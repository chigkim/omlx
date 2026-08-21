import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
BASE = (ROOT / "omlx/admin/templates/base.html").read_text(encoding="utf-8")
LOGIN = (ROOT / "omlx/admin/templates/login.html").read_text(encoding="utf-8")


def _css_color(stylesheet: str, selector: str, property_name: str) -> str:
    rule = re.search(rf"{selector}\s*\{{([^}}]*)\}}", stylesheet, re.DOTALL)
    assert rule is not None
    color = re.search(
        rf"{re.escape(property_name)}:\s*(#[0-9a-fA-F]{{6}})", rule.group(1)
    )
    assert color is not None
    return color.group(1)


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    first_luminance = _relative_luminance(first)
    second_luminance = _relative_luminance(second)
    lighter = max(first_luminance, second_luminance)
    darker = min(first_luminance, second_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def test_focus_ring_uses_theme_aware_two_pixel_outline():
    focus_rule = re.search(r":focus-visible\s*\{([^}]*)\}", BASE, re.DOTALL)
    assert focus_rule is not None
    assert "outline: 2px solid var(--focus-ring-color) !important" in focus_rule.group(
        1
    )
    assert "var(--text-primary" not in focus_rule.group(1)


def test_focus_ring_contrasts_with_login_backgrounds():
    light_ring = _css_color(BASE, r":root", "--focus-ring-color")
    dark_ring = _css_color(BASE, r'\[data-theme="dark"\]', "--focus-ring-color")
    dark_page = _css_color(LOGIN, r'\[data-theme="dark"\] body', "background-color")
    dark_control = _css_color(
        LOGIN, r'\[data-theme="dark"\] \.bg-neutral-50', "background-color"
    )

    assert _contrast_ratio(light_ring, "#ffffff") >= 3
    assert _contrast_ratio(dark_ring, dark_page) >= 3
    assert _contrast_ratio(dark_ring, dark_control) >= 3


def _switch_button_tags(template: Path) -> list[tuple[int, str]]:
    """Return (line number, opening tag) for every custom toggle switch."""
    source = template.read_text(encoding="utf-8")
    tags = []
    for match in re.finditer(r"<button\b[^>]*>", source):
        tag = match.group(0)
        if "w-11 h-6" not in tag:
            continue
        tags.append((source.count("\n", 0, match.start()) + 1, tag))
    return tags


def test_every_toggle_switch_exposes_state_and_name():
    unlabeled = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        for line, tag in _switch_button_tags(template):
            location = f"{template.relative_to(ROOT)}:{line}"
            if "x-a11y-switch" not in tag:
                unlabeled.append(f"{location} missing x-a11y-switch")
            elif "aria-label" not in tag:
                unlabeled.append(f"{location} missing aria-label")

    assert not unlabeled, "toggle switches without accessible state/name:\n" + "\n".join(
        unlabeled
    )


STATE_ANNOTATIONS = (
    "x-a11y-pressed",
    "aria-pressed",
    "x-a11y-switch",
    "aria-checked",
    "aria-current",
    "aria-expanded",
    "aria-selected",
)

# Conditions that drive styling without representing selection state, so the
# control needs no aria state. Keyed by the condition rather than by line so the
# list survives edits above it.
NON_STATE_CONDITIONS = (
    # The benchmark card mirrors the checked state of the checkbox it contains.
    "accBenchmarks[b.key]",
    # The prompt profile row mirrors the pressed state of the button it contains.
    "activePromptProfile === p.name",
    # Transient "Copied!" feedback on copy-to-clipboard buttons.
    "copied",
    "wiredLimitCopied",
    # Hover-only restyling; these buttons swap their visible label instead.
    "hover",
    # Toggles that rename themselves (title/label) rather than expose pressed.
    "chat.pinned",
    "micActive()",
    "chatSettings.webSearchEnabled",
    # Styling that mirrors an actual `disabled` attribute.
    "promptDirty && activePromptProfile",
    "importingMtplx",
    "aneTuning.running",
    "globalSettings.model.model_dirs.length > 1",
    # Progress/result feedback on a one-shot action button.
    "uploadTokenValidated",
    # Two-step destructive actions swap their visible label to request
    # confirmation; they are not persistent pressed/selected controls.
    "confirmUnpairFor === device.node_id",
    "confirmUnloadFor === configuredDeployment()",
    "confirmDeactivateFor === configuredDeployment()",
)


def _opening_tags(source: str):
    """Yield (line, tag, tag name) for elements that may be interactive."""
    for match in re.finditer(r"<(button|a|div|label|span)\b", source):
        index, quote = match.end(), None
        while index < len(source):
            char = source[index]
            if quote:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == ">":
                break
            index += 1
        yield source.count("\n", 0, match.start()) + 1, source[match.start() : index + 1], match.group(1)


def test_selectable_controls_expose_their_state():
    """A control whose :class branches on state must expose that state to AT."""
    unexposed = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = template.read_text(encoding="utf-8")
        for line, tag, name in _opening_tags(source):
            if any(annotation in tag for annotation in STATE_ANNOTATIONS):
                continue
            binding = re.search(r':class="([^"]*)"', tag) or re.search(
                r":class='([^']*)'", tag
            )
            if binding is None or "?" not in binding.group(1):
                continue
            if name not in ("button", "a") and "@click" not in tag:
                continue
            condition = " ".join(binding.group(1).split()).split("?")[0].strip()
            if condition in NON_STATE_CONDITIONS:
                continue
            unexposed.append(
                f"{template.relative_to(ROOT)}:{line} state-styled on `{condition}`"
            )

    assert not unexposed, (
        "controls that restyle on state without exposing it (add an aria state, "
        "or list the condition in NON_STATE_CONDITIONS):\n" + "\n".join(unexposed)
    )


# Pointer-only conveniences: each duplicates an action that keyboard users can
# already reach, so they need no role or tab stop of their own.
POINTER_ONLY_CLICKS = (
    # The chat row is clickable for the mouse; its title is a real button.
    "if (renamingChatId !== chat.id) loadChat(chat.id)",
    # The prompt profile row is clickable for the mouse; its label is a button.
    "selectPromptProfile(p.name)",
    # Timeline dots scroll to a message that is reachable by reading the thread.
    "scrollToMessage(dot.index)",
    # The benchmark card toggles for the mouse; its name is the checkbox.
    "accBenchmarks[b.key] = !accBenchmarks[b.key]",
)


def test_clickable_non_button_elements_are_keyboard_operable():
    """@click on a non-interactive element needs a role and a tab stop."""
    unreachable = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = template.read_text(encoding="utf-8")
        for line, tag, name in _opening_tags(source):
            if name in ("button", "a") or "@click" not in tag:
                continue
            # Overlays, dismiss backdrops and stop-propagation wrappers are not
            # controls; they only mirror an action offered elsewhere.
            if "cursor-pointer" not in tag:
                continue
            if 'role="' in tag and "tabindex=" in tag:
                continue
            handler = re.search(r'@click="([^"]*)"', tag)
            if handler and handler.group(1) in POINTER_ONLY_CLICKS:
                continue
            unreachable.append(f"{template.relative_to(ROOT)}:{line} <{name}> @click")

    assert not unreachable, (
        "clickable elements that keyboard users cannot reach:\n"
        + "\n".join(unreachable)
    )


SETTINGS = ROOT / "omlx/admin/templates/dashboard/_settings.html"

# Each cache limit is edited through a slider that carries the percentage and a
# separate readout that carries the derived size in GB. The slider's own value
# is only the percentage, so without aria-valuetext the GB figure is never
# announced as the slider moves.
CACHE_LIMITS = (
    ("hotCachePercent", "hotCacheSizeGB", "hot_cache"),
    ("cachePercent", "cacheSizeGB", "cold_cache"),
)


def _attributes_of_tag_containing(source: str, needle: str) -> str:
    """Return the opening tag that contains ``needle``."""
    position = source.index(needle)
    start = source.rindex("<", 0, position)
    end = source.index(">", position)
    return source[start:end]


def test_cache_sliders_announce_their_percentage():
    """aria-valuetext, not the native value.

    These sliders run 0-50 and 0-100 but read as a percentage, so the raw
    value a screen reader would otherwise announce does not match the visible
    text. aria-valuetext pins the announcement to what is on screen, including
    the "Off" wording at zero.
    """
    source = SETTINGS.read_text(encoding="utf-8")
    for percent_model, size_expression, _key in CACHE_LIMITS:
        slider = _attributes_of_tag_containing(source, f'x-model.number="{percent_model}"')
        assert 'type="range"' in slider
        valuetext = slider.split(':aria-valuetext="')[1].split('"')[0]
        assert percent_model in valuetext and "%" in valuetext
        # The GB figure belongs to the number input next to it, not here.
        assert size_expression not in valuetext, f"{percent_model} slider restates the GB value"


def test_cache_size_is_a_real_number_input():
    """The GB figure is edited in place by a native spinbutton.

    It used to be a span, then a button that swapped itself for an input. Both
    forms announced no value and no way to change it. A real number input gets
    the spinbutton role, keyboard stepping and value announcement from the
    platform, so none of that has to be hand-rolled with ARIA.
    """
    source = SETTINGS.read_text(encoding="utf-8")
    for _percent_model, size_expression, _key in CACHE_LIMITS:
        editor = _attributes_of_tag_containing(source, f':value="{size_expression}"')
        assert editor.startswith("<input")
        assert 'type="number"' in editor
        # Overriding the native role would cost the editing behaviour it brings.
        assert "role=" not in editor
        # The row's <label> already points at the slider, so this input needs a
        # name of its own; "(GB)" distinguishes it from the percentage.
        assert re.search(r'aria-label="\{\{ t\(.*\) \}\} \(GB\)"', editor), (
            f"{size_expression} spinbutton has no accessible name"
        )


def test_cache_size_inputs_are_always_editable():
    """No mode swap.

    Hiding the editor behind a trigger meant arrow keys landed on a control
    that could not step, and committing a value pulled focus out from under
    the user as the editor closed.
    """
    source = SETTINGS.read_text(encoding="utf-8")
    js = (ROOT / "omlx/admin/static/js/dashboard.js").read_text(encoding="utf-8")
    for _percent_model, size_expression, _key in CACHE_LIMITS:
        editor = _attributes_of_tag_containing(source, f':value="{size_expression}"')
        assert "x-show=" not in editor, f"{size_expression} editor is conditionally shown"
        change = editor.split('@change="')[1].split('"')[0]
        assert "= false" not in change, f"{size_expression} editor closes on change: {change}"
    for state in ("editingHotCache", "editingCache"):
        assert state not in source and state not in js, f"{state} is dead state"



def _subtree_after(source: str, tag: str, name: str) -> str:
    """Return the contents of the element that `tag` opens, nesting included."""
    start = source.index(">", source.index(tag)) + 1
    depth, cursor = 0, start
    while True:
        nested = source.find(f"<{name}", cursor)
        close = source.find(f"</{name}>", cursor)
        if close == -1:
            return source[start:]
        if nested != -1 and nested < close:
            depth += 1
            cursor = nested + 1
            continue
        if depth == 0:
            return source[start:close]
        depth -= 1
        cursor = close + 1

def test_children_presentational_roles_hold_no_interactive_content():
    """A role whose children are presentational hides any control inside it."""
    presentational = ("checkbox", "switch", "radio", "slider", "tab", "option")
    buried = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = template.read_text(encoding="utf-8")
        for line, tag, name in _opening_tags(source):
            role = re.search(r'role="([a-z]+)"', tag)
            if role is None or role.group(1) not in presentational:
                continue
            subtree = _subtree_after(source, tag, name)
            for control in re.finditer(r"<(input|select|textarea|button|a)\b", subtree):
                buried.append(
                    f"{template.relative_to(ROOT)}:{line} role={role.group(1)} "
                    f"hides <{control.group(1)}>"
                )
    assert not buried, (
        "controls pruned from the accessibility tree by their parent role:\n"
        + "\n".join(buried)
    )


def _form_field_tags(source: str):
    """Yield (line, tag, tag name) for every form control."""
    for match in re.finditer(r"<(input|select|textarea)\b", source):
        index, quote = match.end(), None
        while index < len(source):
            char = source[index]
            if quote:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == ">":
                break
            index += 1
        yield (
            source.count("\n", 0, match.start()) + 1,
            source[match.start() : index + 1],
            match.group(1),
        )


def test_labels_next_to_a_field_are_associated_with_it():
    """A <label> that neither wraps nor points at its field names nothing.

    Reading down the page a screen reader speaks the label text and then the
    control, so the pairing sounds right; but tabbing, or jumping by control
    type, announces the control alone and the label is never reached.
    """
    orphaned = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = template.read_text(encoding="utf-8")
        targets = set(re.findall(r'<label[^>]*\bfor="([^"]*)"', source))
        for line, tag, name in _form_field_tags(source):
            kind = re.search(r'type="([a-z]+)"', tag)
            if kind and kind.group(1) in ("hidden", "submit", "button"):
                continue
            if "aria-label" in tag or "placeholder" in tag:
                continue
            identifier = re.search(r'\bid="([^"]*)"', tag)
            if identifier and identifier.group(1) in targets:
                continue
            at = source.index(tag)
            opened = source.rfind("<label", 0, at)
            if opened == -1:
                continue
            closed = source.find("</label>", opened)
            if closed > at:
                continue  # wrapped, so implicitly associated
            if re.search(r"\bfor=", source[opened : source.index(">", opened)]):
                continue  # that label already names a different field
            # Only the label immediately before the field can be its label; any
            # control in between means the label belongs to that one instead.
            if re.search(r'<(input|select|textarea|button)\b|role="', source[closed:at]):
                continue
            orphaned.append(f"{template.relative_to(ROOT)}:{line} <{name}>")

    assert not orphaned, (
        "fields sitting beside an unassociated <label> (add for=/id=, or wrap "
        "the field in the label):\n" + "\n".join(orphaned)
    )


def test_settings_sections_are_headings():
    """Section cards must be headings so they can be skipped between."""
    source = SETTINGS.read_text(encoding="utf-8")
    section_labels = re.findall(r"t\('settings\.[a-z.]*section_label'\)", source)
    assert len(section_labels) >= 12
    for match in re.finditer(
        r"<(\w+)[^>]*>\{\{ t\('settings\.[a-z.]*section_label'\) \}\}", source
    ):
        assert match.group(1) in ("h2", "h3", "h4"), (
            f"settings.…section_label rendered in <{match.group(1)}>, not a heading"
        )


def _without_scripts(source: str) -> str:
    """Blank out <script>/<style> bodies, keeping offsets, before scanning markup.

    Their contents are JavaScript and CSS: a string like '<svg' in there is not
    an element and must not be read as one.
    """
    def blank(match: "re.Match[str]") -> str:
        return match.group(1) + re.sub(r"[^\n]", " ", match.group(2)) + match.group(3)

    return re.sub(
        r"(<(?:script|style)\b[^>]*>)(.*?)(</(?:script|style)>)",
        blank,
        source,
        flags=re.DOTALL,
    )


def _button_tags(source: str):
    """Yield (line, opening tag, inner content) for every <button> in a template."""
    source = _without_scripts(source)
    for match in re.finditer(r"<button\b", source):
        index = match.end()
        quote = None
        while index < len(source):
            char = source[index]
            if quote:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == ">":
                break
            index += 1
        end = source.find("</button>", index)
        if end == -1:
            continue
        line = source[: match.start()].count("\n") + 1
        yield line, source[match.start() : index + 1], source[index + 1 : end]


# Buttons whose only content is an icon. Their name has to be spelled out; an
# <svg> or a lucide <i> announces nothing at all.
ICON_ONLY = re.compile(r"^\s*(?:<svg\b|<i\s+data-lucide|<span\b[^>]*>\s*</span>|</?template\b)")


def test_icon_only_buttons_have_an_accessible_name():
    """A button holding nothing but an icon needs aria-label or title.

    The copy buttons across the status and models tables were exactly this: a
    clipboard glyph and no text, so a screen reader reached "button" and had
    nothing else to say about it.
    """
    nameless = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = template.read_text(encoding="utf-8")
        for line, attributes, body in _button_tags(source):
            if re.search(r"x-text|aria-label|aria-labelledby|title=", attributes):
                continue
            # Any literal text or bound text in the body names the button.
            stripped = re.sub(r"<[^>]*>", " ", body)
            if stripped.strip() or "x-text" in body:
                continue
            nameless.append(f"{template.relative_to(ROOT / 'omlx/admin/templates')}:{line}")
    assert not nameless, "icon-only buttons with no accessible name: " + ", ".join(nameless)


def test_copying_is_announced_not_just_coloured():
    """The copy buttons confirm by turning green for two seconds.

    Colour is not available to a screen reader, so the confirmation also goes
    through a polite live region.
    """
    assert 'id="a11y-announcer"' in BASE
    assert 'aria-live="polite"' in BASE
    assert 'role="status"' in BASE
    assert "sr-only" in BASE, "the announcer must not take up layout space"
    assert "window.announce" in BASE

    script = (ROOT / "omlx/admin/static/js/dashboard.js").read_text(encoding="utf-8")
    assert "_announceCopied()" in script
    # Both the clipboard API path and the execCommand fallback announce.
    assert script.count("this._announceCopied();") == 2
    assert "js.success.copied" in script


def test_hover_revealed_controls_are_visible_on_keyboard_focus():
    """opacity-0 until group-hover hides the control from a keyboard user.

    Tabbing lands focus on something that cannot be seen. Tailwind is prebuilt
    in this repo, so the reveal is a stylesheet rule rather than a utility.
    """
    assert ".opacity-0:focus-visible" in BASE
    assert ".opacity-0:focus-within" in BASE


def test_hover_swapped_labels_keep_a_stable_accessible_name():
    """The load/unload pills change their visible text on hover.

    A pointer user sees "Loaded" become "Unload"; nothing swaps for a keyboard
    or screen reader user, so the name has to state both on its own.
    """
    source = SETTINGS.read_text(encoding="utf-8")
    # Focus reveals the swap for sighted keyboard users too.
    assert '@focusin="hover = true"' in source
    assert '@focusout="hover = false"' in source
    for action, state in (("unloadModel", "status_loaded"), ("loadModel", "status_ready")):
        element = _attributes_of_tag_containing(source, f'@click="{action}(model.id)"')
        assert "aria-label" in element, f"{action} pill has no stable name"
        assert state in element, f"{action} pill drops the current state from its name"


def test_every_opening_tag_is_terminated():
    """A missing ">" swallows the next element into the attribute list.

    The model-name copy button in the settings table shipped this way: the
    parser read "<svg" as an attribute, so the icon never rendered at all.
    """
    unterminated = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = _without_scripts(template.read_text(encoding="utf-8"))
        for match in re.finditer(r"<[a-zA-Z][-\w]*", source):
            index = match.end()
            quote = None
            while index < len(source):
                char = source[index]
                if quote:
                    if char == quote:
                        quote = None
                elif char in "\"'":
                    quote = char
                elif char == ">":
                    break
                elif char == "<":
                    line = source[: match.start()].count("\n") + 1
                    name = template.relative_to(ROOT / "omlx/admin/templates")
                    unterminated.append(f"{name}:{line} <{match.group()[1:]}")
                    break
                index += 1
    assert not unterminated, "unterminated opening tags: " + ", ".join(unterminated)


# The load/unload pills are the deliberate exception: their visible text depends
# on hover, so the name has to restate it. Everything else with visible text
# must not carry a label that replaces it.
LABEL_OVER_TEXT_ALLOWED = ("unloadModel(model.id)", "loadModel(model.id)")


def test_aria_label_never_replaces_a_buttons_visible_text():
    """aria-label wins over the text inside the element.

    So adding one to a button that already reads "Cancel" makes it announce
    something else entirely. Only icon-only buttons should carry a label.
    """
    shadowed = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = template.read_text(encoding="utf-8")
        for line, attributes, body in _button_tags(source):
            if "aria-label" not in attributes:
                continue
            if any(allowed in attributes for allowed in LABEL_OVER_TEXT_ALLOWED):
                continue
            visible = re.sub(r"<[^>]*>", "", body).strip()
            if visible or "x-text" in body:
                name = template.relative_to(ROOT / "omlx/admin/templates")
                shadowed.append(f"{name}:{line}")
    assert not shadowed, "aria-label shadowing visible button text: " + ", ".join(shadowed)


def test_no_attributes_stranded_outside_their_element():
    """An attribute placed after the tag's closing '>' is rendered as text.

    It silently stops being an attribute -- an aria-label written there names
    nothing and shows up on screen as literal markup instead.
    """
    stranded = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = _without_scripts(template.read_text(encoding="utf-8"))
        lines = source.split("\n")
        for index in range(1, len(lines)):
            line = lines[index]
            previous = lines[index - 1].rstrip()
            if not re.match(r'^\s*:?[a-zA-Z-]+="[^"]*"\s*$', line):
                continue
            if previous.endswith(">") and not previous.endswith("/>"):
                name = template.relative_to(ROOT / "omlx/admin/templates")
                stranded.append(f"{name}:{index + 1} {line.strip()[:60]}")
    assert not stranded, "attributes stranded in element content: " + ", ".join(stranded)


# Pre-existing hardcoded labels on plain <div>s in the cluster view. They are
# ignored by assistive technology, but fixing them means deciding what those
# regions are, which is out of scope here.
UNROLED_LABELS_ALLOWED = ("Live cluster measurements", "Memory legend")


def test_aria_label_is_not_put_on_an_element_that_cannot_carry_one():
    """aria-label on a generic element is dropped by assistive technology.

    Only elements with a semantic role expose it, so a label on a bare div or
    span is silently useless -- and usually a sign it landed on the wrong node,
    such as a modal backdrop instead of the close button inside it.
    """
    ignored = []
    for template in sorted((ROOT / "omlx/admin/templates").rglob("*.html")):
        source = _without_scripts(template.read_text(encoding="utf-8"))
        for line, tag, name in _opening_tags(source):
            if name not in ("div", "span", "p", "li", "i", "small"):
                continue
            if not re.search(r"\s:?aria-label=", tag):
                continue
            if 'role="' in tag or "tabindex=" in tag:
                continue
            if any(allowed in tag for allowed in UNROLED_LABELS_ALLOWED):
                continue
            ignored.append(f"{template.relative_to(ROOT)}:{line} <{name}>")

    assert not ignored, "aria-label on an element with no role: " + ", ".join(ignored)

from typing import Any, Dict, List


def parse_blocks_to_outline(blocks: List[Dict[str, Any]]) -> str:
    """Converts Notion block JSON items into a concise, token-efficient structural outline.

    Extracts block types, textual content, and block IDs so the Planner LLM has complete
    structural awareness of the target page without suffering from token bloat.
    """
    if not blocks:
        return "(Page is currently empty - no existing blocks)"

    lines: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue

        b_type = b.get("type", "")
        b_id = b.get("id", "")
        content_dict = b.get(b_type, {}) if isinstance(b.get(b_type), dict) else {}

        # Extract plain text from rich_text array
        rich_text = content_dict.get("rich_text", [])
        text = ""
        if isinstance(rich_text, list):
            text = "".join(
                t.get("plain_text", "")
                for t in rich_text
                if isinstance(t, dict) and "plain_text" in t
            ).strip()

        # Format depending on block type
        if b_type == "heading_1":
            lines.append(f"# {text or 'Untitled'} (ID: `{b_id}`)")
        elif b_type == "heading_2":
            lines.append(f"## {text or 'Untitled'} (ID: `{b_id}`)")
        elif b_type == "heading_3":
            lines.append(f"### {text or 'Untitled'} (ID: `{b_id}`)")
        elif b_type == "to_do":
            checked = "[x]" if content_dict.get("checked") else "[ ]"
            lines.append(f"  {checked} {text} (ID: `{b_id}`)")
        elif b_type == "bulleted_list_item":
            lines.append(f"  • {text} (ID: `{b_id}`)")
        elif b_type == "numbered_list_item":
            lines.append(f"  1. {text} (ID: `{b_id}`)")
        elif b_type == "toggle":
            lines.append(f"  ▶ [toggle] {text or 'Toggle'} (ID: `{b_id}`)")
        elif b_type == "child_page":
            title = content_dict.get("title", "Untitled Subpage")
            lines.append(f"  📄 [child_page] \"{title}\" (ID: `{b_id}`)")
        elif b_type == "child_database":
            title = content_dict.get("title", "Untitled Database")
            lines.append(f"  🗄️ [database] \"{title}\" (ID: `{b_id}`)")
        elif b_type == "callout":
            lines.append(f"  💡 [callout] {text} (ID: `{b_id}`)")
        elif b_type == "divider":
            lines.append(f"  --- (divider ID: `{b_id}`)")
        elif b_type == "paragraph":
            if text:
                preview = text[:80] + ("..." if len(text) > 80 else "")
                lines.append(f"  [p] {preview} (ID: `{b_id}`)")
        else:
            if text:
                lines.append(f"  [{b_type}] {text[:60]} (ID: `{b_id}`)")
            elif b_id:
                lines.append(f"  [{b_type}] (ID: `{b_id}`)")

    return "\n".join(lines) if lines else "(Page has no readable text blocks)"

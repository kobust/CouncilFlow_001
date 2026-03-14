"""
Mayor's Communication / motions schema bundle: JSON shape + transformers to Markdown.

Expects JSON: { mayors_communication_date, computed_docket_date, accounts_master_list, motions[] }.
Transformers: Table (motions as Markdown table), Minutes (annotated committee minutes).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from output_schemas import register_schema

# Committee order for minutes (strict, from prompt)
COMMITTEE_ORDER = [
    "Committee on Personnel, Veterans and Human Services",
    "Committee on Transportation and Traffic",
    "Committee on Zoning and Land Use",
    "Committee on IT and Infrastructure",
    "Committee on Public Safety and Emergency Management",
    "Committee on Public Works",
    "Committee on City Property and Claims",
    "Committee on Licenses",
    "Committee on Ordinances, Elections and Legislative Matters",
    "Committee on Finance",
]

VOTE_REQUIREMENT_TO_DISPLAY = {
    "RC-MAJ": "ROLL CALL - MAJORITY",
    "RC-2/3rds": "ROLL CALL - 2/3 MAJORITY",
    "RC-3/4ths": "ROLL CALL - 3/4 MAJORITY",
    "VV": "VOICE VOTE - MAJORITY",
    "NR": "NOT REQUIRED",
    "UNKNOWN": "UNKNOWN",
}


def _normalize_line_endings(s: str) -> str:
    """Normalize \\r\\n and \\r to \\n so line breaks are consistent."""
    if not isinstance(s, str):
        return str(s) if s is not None else ""
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _to_md_line_breaks(s: str) -> str:
    """Convert newlines to Markdown line breaks (two trailing spaces + newline). Renders as visible line breaks."""
    if not isinstance(s, str):
        return str(s) if s is not None else ""
    norm = _normalize_line_endings(s)
    # Two trailing spaces before newline = Markdown soft line break
    return norm.replace("\n", "  \n")


def _escape_dollars(s: str) -> str:
    """Escape dollar signs to prevent LaTeX math interpretation in Markdown."""
    if not isinstance(s, str):
        return str(s) if s is not None else ""
    return s.replace("$", "\\$")


def _escape_table_cell(s: str, max_len: int | None = 80) -> str:
    """Escape pipes and dollar signs for Markdown table; convert newlines to <br>. max_len=None = no truncation."""
    if not isinstance(s, str):
        return str(s) if s is not None else ""
    # Use <br> to preserve line breaks in table cells (Markdown tables are single-line per row)
    out = s.replace("|", "\\|").replace("\n", "<br>").replace("$", "\\$")
    if max_len is not None and len(out) > max_len:
        return out[: max_len - 3] + "..."
    return out


def _format_long_date(date_str: str | None) -> str:
    """Convert YYYY-MM-DD to long format (e.g. January 20, 2026)."""
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str.strip()[:10], "%Y-%m-%d")
        return dt.strftime("%B %d, %Y")
    except (ValueError, TypeError):
        return date_str

MAYORS_COMMUNICATION_SCHEMA = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://cityofattleboro.us/schemas/mayors-communication-canonical.schema.json",
  "title": "Mayor's Communication Canonical Record",
  "type": "object",
  "additionalProperties": False,
  "required": [
    "mayors_communication_date",
    "computed_docket_date",
    "accounts_master_list",
    "motions"
  ],
  "properties": {
    "mayors_communication_date": {
      "type": "string",
      "format": "date",
      "description": "Date printed on the Mayor's Communication."
    },
    "computed_docket_date": {
      "type": "string",
      "format": "date",
      "description": "Next full regular Municipal Council meeting after the Mayor's Communication date."
    },
    "accounts_master_list": {
      "type": "array",
      "description": "All accounts mentioned in the Mayor's Communication. Unique entries only.",
      "items": { "type": "string", "minLength": 1 },
      "uniqueItems": True
    },
    "motions": {
      "type": "array",
      "description": "One object per numbered item in the Mayor's Communication.",
      "minItems": 1,
      "items": { "$ref": "#/$defs/motion_item" }
    }
  },
  "$defs": {
    "committee_name": {
      "type": "string",
      "enum": [
        "Committee on Personnel, Veterans and Human Services",
        "Committee on Transportation and Traffic",
        "Committee on Zoning and Land Use",
        "Committee on IT and Infrastructure",
        "Committee on Public Safety and Emergency Management",
        "Committee on Public Works",
        "Committee on City Property and Claims",
        "Committee on Licenses",
        "Committee on Ordinances, Elections and Legislative Matters",
        "Committee on Finance",
        "FYI"
      ],
      "description": "Must be exactly one of the standing committee names (or FYI)."
    },

    "vote_requirement_code": {
      "type": "string",
      "enum": ["RC-MAJ", "RC-2/3rds", "RC-3/4ths", "VV", "NR", "UNKNOWN"],
      "description": "Voting requirement code from the Voting Index (or UNKNOWN if uncategorizable)."
    },

    "public_hearing_value": {
      "type": "string",
      "enum": ["Yes", "No", "Yes - Joint", "UNKNOWN"]
    },

    "advertising_value": {
      "type": "string",
      "pattern": "^(NR|[0-9]{1,3}d|\\?\\?\\?)$"
    },

    "citation": {
      "type": "object",
      "description": "Legal or reference citation associated with this motion item.",
      "additionalProperties": True
    },

    "motion_item": {
      "type": "object",
      "additionalProperties": False,
      "required": [
        "mc_item_number",
        "item_id",
        "raw_item_text_verbatim",
        "description_clean",
        "committee_assignment",
        "vote_category_lookup",
        "vote_requirement",
        "public_hearing",
        "advertising_requirement",
        "draft_motion_text"
      ],
      "properties": {
        "mc_item_number": {
          "type": "integer",
          "minimum": 1,
          "description": "Mayor's Communication numbered item."
        },

        "item_id": {
          "type": "string",
          "pattern": "^[0-9]{2}\\.[0-9]{2}\\.[0-9]{2}-MC-[0-9]{3}$",
          "description": "YY.MM.DD-MC-### (docket date derived at top-level)."
        },

        "raw_item_text_verbatim": {
          "type": "string",
          "minLength": 1,
          "description": "verbatim block between number and REFERRED TO COMMITTEE"
        },

        "description_clean": {
          "type": "string",
          "minLength": 1,
          "description": "Short normalized summary of the requested action (prose only)."
        },

        "committee_assignment": { "$ref": "#/$defs/committee_name" },

        "vote_category_lookup": {
          "type": "string",
          "minLength": 1,
          "description": "Exact 'Description' value from the Advertising and Voting Requirements Index (or UNKNOWN)."
        },

        "vote_requirement": { "$ref": "#/$defs/vote_requirement_code" },

        "vote_type_display": {
          "type": ["string", "null"],
          "description": "Optional: pre-rendered vote type string (e.g., 'ROLL CALL - MAJORITY'). If provided, must match translation rules."
        },

        "public_hearing": { "$ref": "#/$defs/public_hearing_value" },

        "advertising_requirement": { "$ref": "#/$defs/advertising_value" },

        "draft_motion_text": {
          "type": ["string", "null"],
          "description": "Text after 'I hereby request your honorable body to' (do not include the leading phrase)."
        },

        "citations": {
          "type": "array",
          "items": { "$ref": "#/$defs/citation" }
        },

        "analysis_block": {
          "type": "object",
          "additionalProperties": False,
          "description": "Optional. Populated by analysis injection step.",
          "required": ["analysis_context", "potential_questions", "include_in_minutes"],
          "properties": {
            "analysis_context": { "type": ["string", "null"] },
            "potential_questions": {
              "type": "array",
              "items": { "type": "string", "minLength": 1 }
            },
            "include_in_minutes": {
              "type": "boolean",
              "default": True,
              "description": "Set false for routine transfers (omit analysis block in minutes)."
            }
          }
        },

        "rendering_hints": {
          "type": "object",
          "additionalProperties": False,
          "description": "Optional renderer hints (must never change literals).",
          "properties": {
            "exclude_from_minutes": {
              "type": "boolean",
              "default": False,
              "description": "Set true for FYI items (Step 3 excludes FYI)."
            }
          }
        }
      }
    },

    "error_item": {
      "type": "object",
      "additionalProperties": False,
      "required": ["code", "message"],
      "properties": {
        "code": {
          "type": "string",
          "enum": [
            "ACCOUNT_EXTRACTION_AMBIGUOUS",
            "ACCOUNT_MISMATCH",
            "VOTE_METADATA_MISSING",
            "SCHEMA_VALIDATION_FAILED"
          ]
        },
        "message": { "type": "string", "minLength": 1 },
        "item_id": { "type": ["string", "null"] },
        "page": { "type": ["integer", "null"], "minimum": 1 }
      }
    }
  }
}

PUBLIC_TESTIMONY_OUTPUT_SCHEMA = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://attleboro-ma.gov/schemas/public-testimony-agenda-output-wrapped-with-date.schema.json",
  "title": "Attleboro Municipal Council Written Public Testimony Output (Agenda-Based, Wrapped, With Date)",
  "description": "Schema for structured extraction of written public testimony matched to matters listed on a Council agenda or to other municipal issues.",
  "type": "object",
  "additionalProperties": False,
  "required": ["items"],
  "properties": {
    "items": {
      "type": "array",
      "description": "One output record per testimony file processed.",
      "items": {
        "$ref": "#/$defs/testimonyItem"
      }
    }
  },
  "$defs": {
    "testimonyItem": {
      "type": "object",
      "additionalProperties": False,
      "required": [
        "filename",
        "author",
        "how-to-contact",
        "date",
        "summary",
        "position",
        "categorization"
      ],
      "properties": {
        "filename": {
          "type": "string",
          "minLength": 1,
          "description": "Filename of the input testimony document or email."
        },
        "author": {
          "type": "string",
          "minLength": 1,
          "description": "Name of the author, or 'Anonymous' if the author cannot be determined."
        },
        "how-to-contact": {
          "type": "string",
          "minLength": 1,
          "description": "Contact information explicitly present in the source, or 'Not provided'."
        },
        "date": {
          "type": "string",
          "description": "Date explicitly available in the source, normalized when possible. Allowed forms: YYYY-MM-DD, YYYY-MM, YYYY, or 'Not provided'.",
          "anyOf": [
            {
              "pattern": "^\\d{4}-\\d{2}-\\d{2}$"
            },
            {
              "pattern": "^\\d{4}-\\d{2}$"
            },
            {
              "pattern": "^\\d{4}$"
            },
            {
              "const": "Not provided"
            }
          ]
        },
        "summary": {
          "type": "string",
          "minLength": 1,
          "description": "Brief neutral summary of the testimony."
        },
        "position": {
          "type": "string",
          "enum": [
            "In Favor",
            "Opposed",
            "Neither For Nor Against"
          ],
          "description": "Constituent position relative to the matched agenda item or categorized issue."
        },
        "categorization": {
          "type": "string",
          "description": "Primary categorization of the testimony.",
          "anyOf": [
            {
              "pattern": "^Agenda Item \\[[^\\]]+\\] - .+$",
              "description": "Matched to a specific agenda item, including items with a listed ID or the literal label 'No listed ID'."
            },
            {
              "pattern": "^Ward Issue - .+$",
              "description": "Localized constituent or neighborhood issue not clearly tied to a current agenda item."
            },
            {
              "pattern": "^General Municipal Issue - .+$",
              "description": "Citywide or general policy issue not clearly tied to a current agenda item."
            },
            {
              "const": "Not Within Council Jurisdiction / Unclear",
              "description": "Submission is unrelated, too vague, or not reasonably connected to Council business."
            }
          ]
        }
      }
    }
  },
  "examples": [
    {
      "items": [
        {
          "filename": "BusinessesinAttleboro.docx",
          "author": "Julie Hall",
          "how-to-contact": "Not provided",
          "date": "2026-02-16",
          "summary": "The author supports opposition to a proposed tax break for a development at 61 Union Street and argues that all businesses should be treated fairly under city ordinances.",
          "position": "Opposed",
          "categorization": "Agenda Item [No listed ID] - Tax increment financing agreement for 61 Union Street"
        }
      ]
    }
  ]
}


def to_public_testimony_table_md(data: Any) -> str:
    """
    Transform public testimony JSON (wrapped object with "items" array of testimonyItem) to a Markdown table.
    Columns: filename, author, how-to-contact, date, position, categorization, summary.
    """
    if not isinstance(data, dict) or "items" not in data:
        return f"Unexpected shape: expected object with 'items' array, got {type(data).__name__}"

    items = data["items"]
    if not isinstance(items, list):
        return f"Unexpected shape: 'items' must be an array, got {type(items).__name__}"

    if not items:
        return "## Public Testimony\n\nNo testimony items."

    parts: list[str] = []

    # Table header
    parts.append("| Filename | Author | How to contact | Date | Position | Categorization | Summary |")
    parts.append("|----------|--------|----------------|------|----------|----------------|---------|")

    for item in items:
        if not isinstance(item, dict):
            continue

        filename = _escape_table_cell(str(item.get("filename", "")), max_len=None)
        author = _escape_table_cell(str(item.get("author", "")), max_len=None)
        how_to_contact = _escape_table_cell(str(item.get("how-to-contact", "")), max_len=None)
        date = _escape_table_cell(str(item.get("date", "")), max_len=None)
        position = _escape_table_cell(str(item.get("position", "")), max_len=None)
        categorization = _escape_table_cell(str(item.get("categorization", "")), max_len=None)
        summary = _escape_table_cell(str(item.get("summary", "")), max_len=None)

        row = (
            f"| {filename} | {author} | {how_to_contact} | {date} | "
            f"{position} | {categorization} | {summary} |"
        )
        parts.append(row)

    return "\n".join(parts)


# Order for sorting testimony by position within a category
_PUBLIC_TESTIMONY_POSITION_ORDER = ("In Favor", "Opposed", "Neither For Nor Against")


def to_public_testimony_by_category_md(data: Any) -> str:
    """
    Transform public testimony JSON into markdown grouped by category, then by position
    (In Favor, Opposed, Neither For Nor Against). Each category has a short blurb with
    counts. Output is plain markdown suitable for export to Google Docs.
    """
    if not isinstance(data, dict) or "items" not in data:
        return f"Unexpected shape: expected object with 'items' array, got {type(data).__name__}"

    items = data["items"]
    if not isinstance(items, list):
        return f"Unexpected shape: 'items' must be an array, got {type(items).__name__}"

    if not items:
        return "## Public Testimony\n\nNo testimony items."

    # Collect valid items and group by categorization
    by_category: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cat = str(item.get("categorization", "")).strip()
        if not cat:
            cat = "Uncategorized"
        by_category.setdefault(cat, []).append(item)

    # Sort categories by first occurrence order (preserve logical order)
    category_order = list(by_category.keys())

    def position_sort_key(it: dict[str, Any]) -> int:
        pos = str(it.get("position", ""))
        try:
            return _PUBLIC_TESTIMONY_POSITION_ORDER.index(pos)
        except ValueError:
            return len(_PUBLIC_TESTIMONY_POSITION_ORDER)

    parts: list[str] = ["# Public Testimony\n"]

    for cat in category_order:
        group = by_category[cat]
        # Sort by position: In Favor, Opposed, Neither For Nor Against
        group = sorted(group, key=position_sort_key)

        in_favor = sum(1 for it in group if it.get("position") == "In Favor")
        opposed = sum(1 for it in group if it.get("position") == "Opposed")
        neither = sum(1 for it in group if it.get("position") == "Neither For Nor Against")

        parts.append(f"## {cat}\n")
        blurb_parts = [f"{len(group)} piece{'s' if len(group) != 1 else ''} of public testimony"]
        if in_favor or opposed or neither:
            count_bits = []
            if in_favor:
                count_bits.append(f"{in_favor} in favor")
            if opposed:
                count_bits.append(f"{opposed} opposed")
            if neither:
                count_bits.append(f"{neither} neither for nor against")
            blurb_parts.append(": " + ", ".join(count_bits) + ".")
        else:
            blurb_parts.append(".")
        parts.append("".join(blurb_parts) + "\n")

        for it in group:
            author = str(it.get("author", "")).strip() or "—"
            date_val = str(it.get("date", "")).strip()
            position = str(it.get("position", "")).strip() or "—"
            summary = str(it.get("summary", "")).strip() or "—"
            filename = str(it.get("filename", "")).strip()

            if date_val and date_val.lower() != "not provided":
                author_line = f"**{author}** ({date_val}) — *{position}*"
            else:
                author_line = f"**{author}** — *{position}*"
            if filename:
                author_line += f" — *{filename}*"
            parts.append(author_line)
            parts.append("")
            parts.append(summary)
            parts.append("")
            parts.append("")

    return "\n".join(parts).strip()


def to_motions_table_md(data: Any) -> str:
    """
    Transform motions JSON to a Markdown table.
    Includes: item_id, description_clean, committee_assignment, vote_type_display, 
    vote_category_lookup, vote_requirement, public_hearing, advertising_requirement, 
    draft_motion_text, analysis (from analysis_block.analysis_context), 
    questions (from analysis_block.potential_questions).
    """
    if not isinstance(data, dict):
        return f"Unexpected shape: expected object, got {type(data).__name__}"

    motions = data.get("motions")
    if not isinstance(motions, list):
        return "Missing or invalid field: motions (expected array)"

    if not motions:
        header = "## Motions\n\n"
        if data.get("mayors_communication_date") or data.get("computed_docket_date"):
            header += f"*Mayor's Communication: {data.get('mayors_communication_date', '—')} · Docket: {data.get('computed_docket_date', '—')}*\n\n"
        return header + "No motions."

    parts: list[str] = []

    # Add header metadata
    mc_date = data.get("mayors_communication_date", "")
    docket_date = data.get("computed_docket_date", "")
    if mc_date or docket_date:
        parts.append(f"*Mayor's Communication: {mc_date} · Docket: {docket_date}*\n")

    # Table header
    parts.append("| Item ID | Description | Committee | Vote Type | Vote Category | Vote Req | Public Hearing | Advertising | Motion Text | Analysis | Questions |")
    parts.append("|---------|-------------|-----------|-----------|---------------|----------|----------------|-------------|-------------|----------|------------|")

    # Table rows
    for m in motions:
        if not isinstance(m, dict):
            continue
        
        # Extract and escape fields (no truncation)
        item_id = _escape_table_cell(m.get("item_id", ""), max_len=None)
        desc = _escape_table_cell(m.get("description_clean", ""), max_len=None)
        committee = _escape_table_cell(m.get("committee_assignment", ""), max_len=None)
        
        # vote_type_display - if provided, use it; otherwise derive from vote_requirement
        vote_type_display = m.get("vote_type_display")
        if vote_type_display:
            vote_type = _escape_table_cell(str(vote_type_display), max_len=None)
        else:
            vote_code = m.get("vote_requirement", "UNKNOWN")
            vote_type = _escape_table_cell(VOTE_REQUIREMENT_TO_DISPLAY.get(vote_code, vote_code), max_len=None)
        
        vote_category = _escape_table_cell(m.get("vote_category_lookup", ""), max_len=None)
        vote_req = _escape_table_cell(m.get("vote_requirement", ""), max_len=None)
        ph = _escape_table_cell(m.get("public_hearing", ""), max_len=None)
        advert = _escape_table_cell(m.get("advertising_requirement", ""), max_len=None)
        
        motion_text = m.get("draft_motion_text") or ""
        motion_str = _escape_table_cell(str(motion_text), max_len=None)

        # Analysis and Questions from analysis_block
        ab = m.get("analysis_block")
        if isinstance(ab, dict):
            analysis_raw = ab.get("analysis_context")
            analysis_str = _escape_table_cell(str(analysis_raw) if analysis_raw is not None else "", max_len=None)
            qs = ab.get("potential_questions")
            if isinstance(qs, list) and qs:
                questions_str = _escape_table_cell(" • ".join(str(q).strip() for q in qs if isinstance(q, str) and q.strip()), max_len=None)
            else:
                questions_str = ""
        else:
            analysis_str = ""
            questions_str = ""

        # Build table row
        row = f"| {item_id} | {desc} | {committee} | {vote_type} | {vote_category} | {vote_req} | {ph} | {advert} | {motion_str} | {analysis_str} | {questions_str} |"
        parts.append(row)

    return "\n".join(parts)


def to_minutes_md(data: Any) -> str:
    """
    Transform canonical JSON to Annotated Committee Minutes markdown.
    Excludes FYI items and motions with exclude_from_minutes or include_in_minutes=false.
    Groups by committee in strict order, sorts by item_id within committee.
    """
    if not isinstance(data, dict):
        return f"Unexpected shape: expected object, got {type(data).__name__}"

    motions = data.get("motions")
    if not isinstance(motions, list):
        return "Missing or invalid field: motions (expected array)"

    accounts_master = set()
    for a in data.get("accounts_master_list") or []:
        if isinstance(a, str) and a.strip():
            accounts_master.add(a.strip())

    docket_date_raw = data.get("computed_docket_date", "")
    docket_date_long = _format_long_date(docket_date_raw)

    def _include_motion(m: Any) -> bool:
        if not isinstance(m, dict):
            return False
        if m.get("committee_assignment") == "FYI":
            return False
        rh = m.get("rendering_hints")
        if isinstance(rh, dict) and rh.get("exclude_from_minutes"):
            return False
        ab = m.get("analysis_block")
        if isinstance(ab, dict) and ab.get("include_in_minutes") is False:
            return False
        return True

    included = [m for m in motions if _include_motion(m)]

    def _committee_sort_key(m: dict) -> tuple[int, str]:
        comm = m.get("committee_assignment", "")
        try:
            idx = COMMITTEE_ORDER.index(comm)
        except ValueError:
            idx = 999
        return (idx, m.get("item_id", ""))

    included.sort(key=_committee_sort_key)

    parts: list[str] = ["# Annotated Committee Minutes", ""]
    last_committee = ""

    for m in included:
        comm = m.get("committee_assignment", "")
        if comm and comm != last_committee:
            parts.append(f"## {comm}")
            parts.append("")
            last_committee = comm
        item_id = m.get("item_id", "")
        draft_text = m.get("draft_motion_text")
        vote_code = m.get("vote_requirement", "UNKNOWN")
        vote_display = VOTE_REQUIREMENT_TO_DISPLAY.get(vote_code, "UNKNOWN")

        parts.append(f"### [{item_id}] FROM THE DOCKET OF: {docket_date_long}")
        parts.append("")

        if draft_text is None or (isinstance(draft_text, str) and not draft_text.strip()):
            motion_line = f"I entertain a motion to TAKE UP ITEM [{item_id}] AS WRITTEN."
        else:
            dt_norm = _normalize_line_endings(draft_text.strip())
            dt_md = _to_md_line_breaks(dt_norm)  # preserve line breaks
            motion_line = f"I entertain a motion to {_escape_dollars(dt_md)}"

        parts.append(motion_line)
        parts.append("")

        ab = m.get("analysis_block")
        if isinstance(ab, dict) and ab.get("include_in_minutes", True):
            ctx = ab.get("analysis_context")
            if ctx is not None and str(ctx).strip():
                # Preserve all lines including blank lines; normalize line endings
                ctx_norm = _normalize_line_endings(str(ctx))
                lines = [_escape_dollars(ln.rstrip()) for ln in ctx_norm.split("\n")]
                if lines:
                    parts.append(f"> **Analysis/context:** {lines[0]}")
                    for ln in lines[1:]:
                        parts.append(f"> {ln}")
            questions = ab.get("potential_questions") if isinstance(ab, dict) else []
            if isinstance(questions, list) and questions:
                parts.append("")
                parts.append("> **Potential Questions:**")
                for q in questions:
                    if isinstance(q, str) and q.strip():
                        # Preserve line breaks within each question; normalize line endings
                        q_norm = _normalize_line_endings(q)
                        q_lines = q_norm.split("\n")
                        parts.append(f"> - {_escape_dollars(q_lines[0].strip())}")
                        for cont in q_lines[1:]:
                            parts.append(f">   {_escape_dollars(cont.strip())}")
            if ctx or (isinstance(questions, list) and questions):
                parts.append("")

        parts.append(vote_display + "\n")
        parts.append("Seconded _____________________ Motion [ ] Passed or [ ] Rejected\n")
        parts.append("YEAS_____ NAYS_____ ABSENT_____ ABSTAIN_____\n")
        parts.append("\n")

    return "\n".join(parts).strip() or "# Annotated Committee Minutes"


def load() -> None:
    register_schema(
        "mayors_communication",
        MAYORS_COMMUNICATION_SCHEMA,
        [
            ("Table", to_motions_table_md),
            ("Minutes", to_minutes_md),
        ],
    )
    register_schema(
        "public_testimony_output",
        PUBLIC_TESTIMONY_OUTPUT_SCHEMA,
        [
            ("Table", to_public_testimony_table_md),
            ("By category", to_public_testimony_by_category_md),
        ],
    )

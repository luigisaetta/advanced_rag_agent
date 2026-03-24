#!/usr/bin/env python3
"""
Apply deployment settings from a single setup.env file to:
- config.py
- config_private.py
- deployment/docker/docker-compose.yml
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = Path(__file__).resolve().with_name("setup.env")


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE file with # comments."""
    data: dict[str, str] = {}
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid line {lineno} in {path}: {raw}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ) and len(value) >= 2:
            value = value[1:-1]
        data[key] = value
    return data


def py_literal(value: str, as_bool: bool = False) -> str:
    """Format value as python literal."""
    if as_bool:
        return (
            "True"
            if str(value).strip().lower() in {"1", "true", "yes", "on"}
            else "False"
        )
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def replace_assignment(text: str, var_name: str, rhs_literal: str) -> tuple[str, int]:
    """Replace a top-level python assignment and return (updated_text, matches)."""
    pattern = re.compile(rf"(?m)^({re.escape(var_name)}\s*=\s*).*$")
    return pattern.subn(rf"\1{rhs_literal}", text, count=1)


def replace_or_warn(
    text: str, pattern: str, repl: str, label: str, warnings: list[str]
) -> str:
    """Regex replace once, else collect warning."""
    updated, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n == 0:
        warnings.append(f"Pattern not found for {label}")
        return text
    return updated


def update_config_py(text: str, env: dict[str, str], warnings: list[str]) -> str:
    """Apply configurable keys to config.py."""
    mapping = [
        ("CFG_AUTH", "AUTH", False),
        ("CFG_LLM_REGION", "LLM_REGION", False),
        ("CFG_EMBED_REGION", "EMBED_REGION", False),
        ("CFG_EMBED_MODEL_ID", "EMBED_MODEL_ID", False),
        ("CFG_COMPARTMENT_ID", "COMPARTMENT_ID", False),
        ("CFG_LLM_MODEL_ID", "LLM_MODEL_ID", False),
        ("CFG_INTENT_MODEL_ID", "INTENT_MODEL_ID", False),
        ("CFG_RERANKER_MODEL_ID", "RERANKER_MODEL_ID", False),
        (
            "CFG_POST_ANSWER_EVALUATION_MODEL_ID",
            "POST_ANSWER_EVALUATION_MODEL_ID",
            False,
        ),
        ("CFG_MAIN_LANGUAGE", "MAIN_LANGUAGE", False),
        ("CFG_ENABLE_TRACING", "ENABLE_TRACING", True),
        ("CFG_LANGFUSE_HOST", "LANGFUSE_HOST", False),
    ]
    out = text
    for env_key, py_key, is_bool in mapping:
        if env_key not in env:
            continue
        new_out, found = replace_assignment(
            out, py_key, py_literal(env[env_key], as_bool=is_bool)
        )
        if found == 0:
            warnings.append(f"Assignment not found in config.py: {py_key}")
        out = new_out

    if "CFG_CITATION_BASE_URL" in env:
        out = replace_or_warn(
            out,
            r'(?ms)^CITATION_BASE_URL\s*=\s*os\.getenv\(\s*"CITATION_BASE_URL".*?\)\s*$',
            f'CITATION_BASE_URL = os.getenv(\n    "CITATION_BASE_URL", {py_literal(env["CFG_CITATION_BASE_URL"])}\n)',
            "config.py:CITATION_BASE_URL",
            warnings,
        )
    return out


def update_config_private_py(
    text: str, env: dict[str, str], warnings: list[str]
) -> str:
    """Apply configurable keys to config_private.py."""
    mapping = [
        ("PRIV_VECTOR_DB_USER", "VECTOR_DB_USER"),
        ("PRIV_VECTOR_DB_PWD", "VECTOR_DB_PWD"),
        ("PRIV_VECTOR_WALLET_PWD", "VECTOR_WALLET_PWD"),
        ("PRIV_VECTOR_DSN", "VECTOR_DSN"),
        ("PRIV_APM_PUBLIC_KEY", "APM_PUBLIC_KEY"),
        ("PRIV_LANGFUSE_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY"),
        ("PRIV_LANGFUSE_SECRET_KEY", "LANGFUSE_SECRET_KEY"),
    ]
    out = text
    for env_key, py_key in mapping:
        if env_key not in env:
            continue
        new_out, found = replace_assignment(out, py_key, py_literal(env[env_key]))
        if found == 0:
            warnings.append(f"Assignment not found in config_private.py: {py_key}")
        out = new_out

    if "PRIV_LOCAL_WALLET_DIR" in env:
        out = replace_or_warn(
            out,
            r"(?m)^(\s*local_wallet\s*=\s*).*$",
            rf'\1{py_literal(env["PRIV_LOCAL_WALLET_DIR"])}',
            "config_private.py:local_wallet",
            warnings,
        )
    return out


def update_compose_yml(text: str, env: dict[str, str], warnings: list[str]) -> str:
    """Apply host mount paths and optional citation base URL in compose."""
    out = text
    if "DOCKER_WALLET_HOST_PATH" in env:
        wallet = env["DOCKER_WALLET_HOST_PATH"]
        out, count = re.subn(
            r"(?m)^(\s*-\s*)([^:\n]+)(:/app/wallet_atp:ro\s*)$",
            rf"\1{wallet}\3",
            out,
        )
        if count == 0:
            warnings.append("No wallet mount found in docker-compose.yml")

    if "DOCKER_CITATIONS_HOST_PATH" in env:
        pages = env["DOCKER_CITATIONS_HOST_PATH"]
        out, count = re.subn(
            r"(?m)^(\s*-\s*)([^:\n]+)(:/data/citations:ro\s*)$",
            rf"\1{pages}\3",
            out,
        )
        if count == 0:
            warnings.append("No citations mount found in docker-compose.yml")

    if "DOCKER_OCI_HOST_PATH" in env:
        oci_dir = env["DOCKER_OCI_HOST_PATH"]
        out, count = re.subn(
            r"(?m)^(\s*-\s*)([^:\n]+)(:/root/\.oci:ro\s*)$",
            rf"\1{oci_dir}\3",
            out,
        )
        if count == 0:
            warnings.append("No OCI mount found in docker-compose.yml")

    if "DOCKER_CITATION_BASE_URL" in env:
        base = env["DOCKER_CITATION_BASE_URL"]
        out = replace_or_warn(
            out,
            r"(?m)^(\s*-\s*CITATION_BASE_URL=\$\{CITATION_BASE_URL:-).*(\}\s*)$",
            rf"\1{base}\2",
            "docker-compose.yml:CITATION_BASE_URL",
            warnings,
        )

    return out


def validate_paths(env: dict[str, str]) -> list[str]:
    """Validate common filesystem path settings."""
    errors = []
    for key in (
        "DOCKER_WALLET_HOST_PATH",
        "DOCKER_CITATIONS_HOST_PATH",
        "DOCKER_OCI_HOST_PATH",
    ):
        value = env.get(key)
        if not value:
            continue
        expanded = Path(os.path.expandvars(value)).expanduser()
        if not expanded.exists():
            errors.append(f"{key} does not exist: {expanded}")
    return errors


def show_diff(path: Path, before: str, after: str) -> None:
    """Print unified diff for file content changes."""
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=str(path),
        tofile=str(path),
        lineterm="",
    )
    print("\n".join(diff))


def write_with_backup(
    path: Path, content: str, backup_suffix: str, no_backup: bool
) -> None:
    """Write file, optionally creating backup."""
    if not no_backup:
        backup_path = path.with_name(path.name + backup_suffix)
        shutil.copy2(path, backup_path)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Apply deployment setup.env values to config.py, config_private.py and docker-compose.yml."
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to setup env file (default: deployment/docker/setup.env).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["config", "private", "compose", "all"],
        default=["all"],
        help="Which files to update (default: all).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes to disk. Without this flag, only prints a dry-run diff.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup file creation when writing.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Backup suffix for written files (default: .bak).",
    )
    parser.add_argument(
        "--no-validate-paths",
        action="store_true",
        help="Skip filesystem existence checks for host path settings.",
    )
    args = parser.parse_args()

    env_file = Path(args.env_file).expanduser().resolve()
    if not env_file.exists():
        print(f"ERROR: setup file not found: {env_file}")
        print("Tip: cp deployment/docker/setup.env.example deployment/docker/setup.env")
        return 2

    env = parse_env_file(env_file)

    if not args.no_validate_paths:
        path_errors = validate_paths(env)
        if path_errors:
            print("ERROR: path validation failed:")
            for err in path_errors:
                print(f"- {err}")
            print("Use --no-validate-paths to bypass checks.")
            return 2

    target_set = set(args.targets)
    if "all" in target_set:
        target_set = {"config", "private", "compose"}

    plan = []
    if "config" in target_set:
        plan.append((ROOT / "config.py", update_config_py))
    if "private" in target_set:
        plan.append((ROOT / "config_private.py", update_config_private_py))
    if "compose" in target_set:
        plan.append(
            (ROOT / "deployment" / "docker" / "docker-compose.yml", update_compose_yml)
        )

    warnings: list[str] = []
    changes = []

    for path, updater in plan:
        if not path.exists():
            if path.name == "config_private.py":
                template = ROOT / "config_private_template.py"
                if template.exists():
                    shutil.copy2(template, path)
                    print(f"Created missing {path} from template {template}")
                else:
                    print(f"ERROR: target file not found: {path}")
                    return 2
            else:
                print(f"ERROR: target file not found: {path}")
                return 2
        before = path.read_text(encoding="utf-8")
        after = updater(before, env, warnings)
        if before != after:
            changes.append((path, before, after))

    if warnings:
        print("WARNINGS:")
        for item in warnings:
            print(f"- {item}")
        print("")
    else:
        print("Validation completed: no errors found.")
        print("")

    if not changes:
        print("No changes detected. Configuration is already up to date.")
        print("No errors found.")
        return 0

    mode = "WRITE" if args.write else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Env file: {env_file}")
    print("")

    for path, before, after in changes:
        print(f"--- Changes for {path}")
        show_diff(path, before, after)
        print("")

    if not args.write:
        print("Dry-run completed successfully. No errors found.")
        print(
            "To apply these changes, run:\n"
            f"python3 deployment/docker/configure.py --env-file {env_file} --write"
        )
        return 0

    for path, _before, after in changes:
        write_with_backup(path, after, args.backup_suffix, args.no_backup)
        print(f"Updated: {path}")

    print("")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

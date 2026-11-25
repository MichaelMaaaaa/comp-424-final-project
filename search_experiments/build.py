"""Generate a submission by resolving development modules, never editing S0."""

import argparse
import ast
from pathlib import Path

from search_experiments.common import (REPO_ROOT, atomic_json, code_manifest, digest,
                                        file_hash, load_json, normalize_variant)


def _internal(module):
    return module is not None and module.startswith("agents.") and module != "agents.agent"


def _definitions(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return names


def build_agent(variant, destination, *, root=REPO_ROOT, book_module=None):
    variant = normalize_variant(variant)
    root, destination = Path(root), Path(destination)
    parsed, ordered, visiting = {}, [], set()

    def visit(module):
        if module in parsed:
            return
        if module in visiting:
            raise ValueError(f"circular build dependency: {module}")
        visiting.add(module)
        path = root.joinpath(*module.split(".")).with_suffix(".py")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        dependencies = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and _internal(node.module):
                dependencies.append(node.module)
            if isinstance(node, ast.Import):
                dependencies.extend(alias.name for alias in node.names if _internal(alias.name))
        for dependency in dependencies:
            visit(dependency)
        visiting.remove(module)
        parsed[module] = tree
        ordered.append(module)

    visit("agents.experiment_agent")
    book_module = book_module or variant.get("book", {}).get("provider")
    if book_module:
        visit(book_module)
    owners = {}
    for module in ordered:
        for name in _definitions(parsed[module]):
            if name in owners:
                raise ValueError(f"top-level name collision {name!r}: {owners[name]} and {module}")
            owners[name] = module

    imports, seen_imports, bodies, futures = [], set(), [], set()
    for module in ordered:
        aliases = {}
        for node in ast.walk(parsed[module]):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _internal(alias.name):
                        if alias.asname is None:
                            raise ValueError("use an alias for a development module import")
                        aliases[alias.asname] = alias.name

        class InlineImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                if node.module == "__future__":
                    futures.update(alias.name for alias in node.names)
                    return None
                if _internal(node.module):
                    assignments = []
                    for alias in node.names:
                        if alias.name == "*":
                            raise ValueError("wildcard development imports cannot be built")
                        if alias.asname and alias.asname != alias.name:
                            assignments.append(ast.Assign(targets=[ast.Name(id=alias.asname, ctx=ast.Store())],
                                                          value=ast.Name(id=alias.name, ctx=ast.Load())))
                    return assignments
                return node

            def visit_Import(self, node):
                names = [alias for alias in node.names if not _internal(alias.name)]
                return ast.Import(names=names) if names else None

            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id in aliases:
                    return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
                return self.generic_visit(node)

        tree = InlineImports().visit(parsed[module])
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                key = ast.dump(node)
                if key not in seen_imports:
                    seen_imports.add(key)
                    imports.append(node)
            elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                      and isinstance(node.value.value, str)):
                bodies.append(node)
    if not any(isinstance(node, ast.ImportFrom) and node.module == "store" for node in imports):
        imports.append(ast.ImportFrom(module="store", names=[ast.alias(name="register_agent")], level=0))
    evaluation_factory = "make_evaluator" if "make_evaluator" in owners else "default_evaluator"
    book_factory = "make_book_hook" if book_module else "None"
    if book_module and "make_book_hook" not in _definitions(parsed[book_module]):
        raise ValueError("B's provider must define make_book_hook(book_config, variant)")
    wrapper = ast.parse(
        "EXPERIMENT_VARIANT = " + repr(variant) + "\n"
        "@register_agent('student_agent')\n"
        "class StudentAgent(ExperimentAgent):\n"
        "    def __init__(self):\n"
        f"        super().__init__(EXPERIMENT_VARIANT, {evaluation_factory}, {book_factory})\n")
    front = [ast.ImportFrom(module="__future__", names=[ast.alias(name=n) for n in sorted(futures)], level=0)] if futures else []
    merged = ast.fix_missing_locations(ast.Module(body=front + imports + bodies + wrapper.body, type_ignores=[]))
    source = ast.unparse(merged) + "\n"
    ast.parse(source, feature_version=(3, 11))
    if len(source.encode()) > 2_000_000:
        raise ValueError("generated submission exceeds 2 MB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    metadata = {"variant": variant, "source_modules": ordered, "source_files": {
        module: file_hash(root.joinpath(*module.split(".")).with_suffix(".py")) for module in ordered},
        "artifact_hash": file_hash(destination), "bytes": destination.stat().st_size,
        "provenance": code_manifest()}
    atomic_json(destination.with_suffix(".meta.json"), metadata)
    return destination, metadata


def build_opponent(kind, destination, seed=0, budget=1.5):
    """Freeze a baseline or embed the actual RNG seed of random/greedy."""
    source_path = REPO_ROOT / ("baselines/ab_baseline_agent.py" if kind == "ab"
                               else f"agents/{kind}_agent.py")
    if kind not in ("ab", "random", "greedy"):
        raise ValueError("opponent must be ab, random, or greedy")
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register_agent":
            node.args = [ast.Constant("student_agent")]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "default_rng"):
            node.args = [ast.Constant(int(seed))]
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TIME_LIMIT" for t in node.targets):
            node.value = ast.Constant(budget)
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    metadata = {"kind": kind, "opponent_rng_seed": seed, "budget": budget,
                "source_hash": file_hash(source_path), "artifact_hash": file_hash(destination)}
    atomic_json(destination.with_suffix(".meta.json"), metadata)
    return destination, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, help="output directory; contains student_agent.py")
    args = parser.parse_args()
    variant = load_json(args.config)
    variant = variant.get("variant", variant)
    path, meta = build_agent(variant, Path(args.output) / "student_agent.py")
    print(path, meta["artifact_hash"])


if __name__ == "__main__":
    main()




def expand_with_ancestors(
    selected_ids: list[str], ancestors_map: dict[str, set[str]]
) -> set[str]:
    """Return the full active HPO set after ancestor back-propagation."""
    active_ids = set(selected_ids)

    for hpo_id in selected_ids:
        active_ids.update(ancestors_map.get(hpo_id, set()))

    return active_ids
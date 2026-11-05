def find_element_by_id(elements, target_id, id_field="id"):
    """
    Searches for an element in a list of dictionaries by its 'id' key.

    Parameters:
    elements (list): List of dictionaries where each dictionary contains an
    'id' key.
    target_id: The id value to search for.
    id_field: The key to search in.

    Returns:
    dict: The dictionary with the matching 'id' or None if no match is found.
    """
    for element in elements:
        if element.get(id_field) == target_id:
            return element
    return None

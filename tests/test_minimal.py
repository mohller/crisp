# minimal test

def test_import():
    import crisp


def test_basic_usage():
    from crisp.photonuclear_cross_sections import PSB_model
    assert PSB_model() is not None
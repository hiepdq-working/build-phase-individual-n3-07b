from src.data.check_splits import validate

SIZES = {"base": 2, "pool": 2, "dev": 1, "val2k": 1, "arm_k": 1}


def good():
    return {"base": {"train/a", "train/b"}, "pool": {"train/c", "train/d"},
            "dev": {"train/e"}, "val2k": {"val/x"}, "rare_val": {"val/x", "val/y"},
            "arm_random": {"train/c"}}


def test_pass():
    assert validate(good(), SIZES) == []


def test_overlap_fails():
    s = good()
    s["pool"] = {"train/a", "train/d"}
    assert validate(s, SIZES)


def test_arm_outside_pool_fails():
    s = good()
    s["arm_random"] = {"train/a"}
    assert validate(s, SIZES)


def test_wrong_size_fails():
    s = good()
    s["base"] = {"train/a"}
    assert validate(s, SIZES)


def test_val_in_train_split_fails():
    s = good()
    s["base"] = {"train/a", "val/b"}
    assert validate(s, SIZES)

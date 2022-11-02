def upDownRightLeftInner():
    return (
        "iiiiudrl",
        [
            [["Inner"], ["Inner"], ["Inner"], ["Inner"]],
            [
                ["Up"],
                ["Down"],
                ["Right"],
                ["Left"],
            ],
        ],
    )


def upDownRightLeftInnerSpecial():
    return (
        "iiiiudrlspec",
        [
            [["Inner"], ["Inner"], ["Inner"], ["Inner"]],
            [
                ["Up"],
                ["Down"],
                ["Right"],
                ["Left"],
            ],
        ],
    )


# Try combiningWitInner?
def upDownRightLeftVis():
    return (
        "vvvvudrl",
        [
            [["Visualized"], ["Visualized"], ["Visualized"], ["Visualized"]],
            [
                ["Up"],
                ["Down"],
                ["Right"],
                ["Left"],
            ],
        ],
    )


def upDownRightLeftVisInner():
    return (
        "iiiivvvvudrl",
        [
            [
                ["Inner"],
                ["Inner"],
                ["Inner"],
                ["Inner"],
                ["Visualized"],
                ["Visualized"],
                ["Visualized"],
                ["Visualized"],
            ],
            [
                ["Up"],
                ["Down"],
                ["Right"],
                ["Left"],
                ["Up"],
                ["Down"],
                ["Right"],
                ["Left"],
            ],
        ],
    )


def upDownInner():
    return (
        "iiud",
        [
            [["Inner"], ["Inner"]],
            [
                ["Up"],
                ["Down"],
            ],
        ],
    )


def upDownVis():
    return (
        "vvud",
        [
            [["Visualized"], ["Visualized"]],
            [
                ["Up"],
                ["Down"],
            ],
        ],
    )


def upDownVisSpecial():
    return (
        "vvudspec",
        [
            [["Visualized"], ["Visualized"]],
            [
                ["Up"],
                ["Down"],
            ],
        ],
    )


def rightLeftInner():
    return (
        "iirl",
        [
            [["Inner"], ["Inner"]],
            [
                ["Right"],
                ["Left"],
            ],
        ],
    )

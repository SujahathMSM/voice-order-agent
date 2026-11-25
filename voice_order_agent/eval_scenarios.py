"""Step 12: evaluation scenarios. Spoken turns + the EXPECTED confirmed order.

Categories cover everything this project fought: simple adds, corrections，
the kottu mangle（accent)， options, pickup normalization, frustration recovery.
Expectations use menu_ids from data/menu.json - the same truth the agent must
reach, expressed in the same vocabulary the scorer compares with.
"""

SCENARIOS = [
    {
        "id": "S01", "category": "simple",
        # cooperative caller: states items unambiguously and answers questions
        "turns": ["Two chicken kottu and one can of Coke, please",
                  "Yes, that's everything",
                  "Six thirty PM",
                  "Yes, that's right"],
        "expected": {"items": {"kottu-chicken": {"qty": 2}, "soda": {"qty": 1}},
                     "pickup": "18:30"},
    },
    {
        "id": "S02", "category": "correction",
        "turns": ["One fish roll and one plain fried rice",
                  "Yes, that's right",
                  "Actually, change the fish rolls to four",
                  "Seven PM",
                  "Yes confirm"],
        "expected": {"items": {"roll-fish": {"qty": 4},
                               "rice-fried-plain": {"qty": 1}},
                     "pickup": "19:00"},
    },
    {
        "id": "S03", "category": "accent-mangle",
        # spoken with the founder's accent: "kottu" historically became "coke"
        "turns": ["Three chicken kottu please",
                  "Yes, kottu, the roti dish",
                  "Eight PM",
                  "That's right"],
        "expected": {"items": {"kottu-chicken": {"qty": 3}}, "pickup": "20:00"},
    },
    {
        "id": "S04", "category": "options",
        "turns": ["Two deviled chicken sandwiches, no mayo, and one egg hopper",
                  "Nine PM",
                  "Yeah confirm"],
        "expected": {"items": {"sandwich-deviled-chicken": {"qty": 2,
                                                            "options": ["no mayo"]},
                               "hopper-egg": {"qty": 1}},
                     "pickup": "21:00"},
    },
    {
        "id": "S05", "category": "merge",
        "turns": ["One egg hopper please",
                  "Actually add ten more egg hoppers",
                  "Eight thirty PM",
                  "Confirm"],
        "expected": {"items": {"hopper-egg": {"qty": 11}}, "pickup": "20:30"},
    },
    {
        "id": "S06", "category": "remove",
        "turns": ["One fish curry, one chicken curry and a plain tea",
                  "Remove the fish curry",
                  "Six PM",
                  "Yes that's everything"],
        "expected": {"items": {"curry-chicken": {"qty": 1},
                               "tea-plain": {"qty": 1}},
                     "pickup": "18:00"},
    },
    {
        "id": "S07", "category": "menu-question",
        "turns": ["What do you have without onions?",
                  "I'll take the fresh salad then",
                  "Seven thirty PM",
                  "Confirm please"],
        "expected": {"items": {"salad-fresh": {"qty": 1}}, "pickup": "19:30"},
    },
    {
        "id": "S08", "category": "pickup-am",
        # caller says morning explicitly - the agent must not assume evening
        "turns": ["One seafood fried rice for tomorrow morning pickup at 9 AM",
                  "Yes confirm"],
        "expected": {"items": {"rice-fried-seafood": {"qty": 1}},
                     "pickup": "09:00"},
    },
    {
        "id": "S09", "category": "frustration",
        "turns": ["One club sandwich",
                  "What? Sorry, say that again",
                  "Okay. And a chicken soup. No onions in the soup... actually no, two chicken soups",
                  "Ten thirty AM",
                  "Yes, confirm it"],
        "expected": {"items": {"sandwich-club": {"qty": 1},
                               "soup-chicken": {"qty": 2}},
                     "pickup": "10:30"},
    },
    {
        "id": "S10", "category": "drinks-upsell",
        "turns": ["One chicken kottu",
                  "Sure, a woodapple juice, no sugar",
                  "Six forty five PM",
                  "Yes that's right"],
        "expected": {"items": {"kottu-chicken": {"qty": 1},
                               "juice-woodapple": {"qty": 1,
                                                   "options": ["no sugar"]}},
                     "pickup": "18:45"},
    },
    {
        "id": "S11", "category": "multi-item-average",
        "turns": ["Two milk rice, one seeni sambol, three samosas and a plain tea",
                  "Eight fifteen PM",
                  "Confirmed"],
        "expected": {"items": {"rice-milk": {"qty": 2},
                               "sambol-seeni": {"qty": 1},
                               "samosa": {"qty": 3},
                               "tea-plain": {"qty": 1}},
                     "pickup": "20:15"},
    },
    {
        "id": "S12", "category": "correction-chain",
        # the hardest: correct one item AND change another's quantity in one go
        "turns": ["Two deviled fish sandwiches and one noodles chicken",
                  "Change the sandwiches to deviled chicken, and make the noodles two",
                  "Nine PM",
                  "Yes confirm"],
        "expected": {"items": {"sandwich-deviled-chicken": {"qty": 2},
                               "noodles-chicken": {"qty": 2}},
                     "pickup": "21:00"},
    },
]

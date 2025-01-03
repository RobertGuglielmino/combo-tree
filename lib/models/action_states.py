ACTION_STATES = ["DEAD_DOWN",
"DEAD_LEFT",
"DEAD_RIGHT",
"DEAD_UP",
"DEAD_UP_STAR",
"DEAD_UP_STAR_ICE",
"DEAD_UP_FALL",
"DEAD_UP_FALL_HIT_CAMERA",
"DEAD_UP_FALL_HIT_CAMERA_FLAT",
"DEAD_UP_FALL_ICE",
"DEAD_UP_FALL_HIT_CAMERA_ICE",
"SLEEP",
"REBIRTH",
"REBIRTH_WAIT",
"WAIT",
"WALK_SLOW",
"WALK_MIDDLE",
"WALK_FAST",
"TURN",
"TURN_RUN",
"DASH",
"RUN",
"RUN_DIRECT",
"RUN_BRAKE",
"KNEE_BEND",
"JUMP_F",
"JUMP_B",
"JUMP_AERIAL_F",
"JUMP_AERIAL_B",
"FALL",
"FALL_F",
"FALL_B",
"FALL_AERIAL",
"FALL_AERIAL_F",
"FALL_AERIAL_B",
"FALL_SPECIAL",
"FALL_SPECIAL_F",
"FALL_SPECIAL_B",
"DAMAGE_FALL",
"SQUAT",
"SQUAT_WAIT",
"SQUAT_RV",
"LANDING",
"LANDING_FALL_SPECIAL",
"ATTACK_11",
"ATTACK_12",
"ATTACK_13",
"ATTACK_100_START",
"ATTACK_100_LOOP",
"ATTACK_100_END",
"ATTACK_DASH",
"ATTACK_S_3_HI",
"ATTACK_S_3_HI_S",
"ATTACK_S_3_S",
"ATTACK_S_3_LW_S",
"ATTACK_S_3_LW",
"ATTACK_HI_3",
"ATTACK_LW_3",
"ATTACK_S_4_HI",
"ATTACK_S_4_HI_S",
"ATTACK_S_4_S",
"ATTACK_S_4_LW_S",
"ATTACK_S_4_LW",
"ATTACK_HI_4",
"ATTACK_LW_4",
"ATTACK_AIR_N",
"ATTACK_AIR_F",
"ATTACK_AIR_B",
"ATTACK_AIR_HI",
"ATTACK_AIR_LW",
"LANDING_AIR_N",
"LANDING_AIR_F",
"LANDING_AIR_B",
"LANDING_AIR_HI",
"LANDING_AIR_LW",
"DAMAGE_HI_1",
"DAMAGE_HI_2",
"DAMAGE_HI_3",
"DAMAGE_N_1",
"DAMAGE_N_2",
"DAMAGE_N_3",
"DAMAGE_LW_1",
"DAMAGE_LW_2",
"DAMAGE_LW_3",
"DAMAGE_AIR_1",
"DAMAGE_AIR_2",
"DAMAGE_AIR_3",
"DAMAGE_FLY_HI",
"DAMAGE_FLY_N",
"DAMAGE_FLY_LW",
"DAMAGE_FLY_TOP",
"DAMAGE_FLY_ROLL",
"LIGHT_GET",
"HEAVY_GET",
"LIGHT_THROW_F",
"LIGHT_THROW_B",
"LIGHT_THROW_HI",
"LIGHT_THROW_LW",
"LIGHT_THROW_DASH",
"LIGHT_THROW_DROP",
"LIGHT_THROW_AIR_F",
"LIGHT_THROW_AIR_B",
"LIGHT_THROW_AIR_HI",
"LIGHT_THROW_AIR_LW",
"HEAVY_THROW_F",
"HEAVY_THROW_B",
"HEAVY_THROW_HI",
"HEAVY_THROW_LW",
"LIGHT_THROW_F_4",
"LIGHT_THROW_B_4",
"LIGHT_THROW_HI_4",
"LIGHT_THROW_LW_4",
"LIGHT_THROW_AIR_F_4",
"LIGHT_THROW_AIR_B_4",
"LIGHT_THROW_AIR_HI_4",
"LIGHT_THROW_AIR_LW_4",
"HEAVY_THROW_F_4",
"HEAVY_THROW_B_4",
"HEAVY_THROW_HI_4",
"HEAVY_THROW_LW_4",
"SWORD_SWING_1",
"SWORD_SWING_3",
"SWORD_SWING_4",
"SWORD_SWING_DASH",
"BAT_SWING_1",
"BAT_SWING_3",
"BAT_SWING_4",
"BAT_SWING_DASH",
"PARASOL_SWING_1",
"PARASOL_SWING_3",
"PARASOL_SWING_4",
"PARASOL_SWING_DASH",
"HARISEN_SWING_1",
"HARISEN_SWING_3",
"HARISEN_SWING_4",
"HARISEN_SWING_DASH",
"STAR_ROD_SWING_1",
"STAR_ROD_SWING_3",
"STAR_ROD_SWING_4",
"STAR_ROD_SWING_DASH",
"LIP_STICK_SWING_1",
"LIP_STICK_SWING_3",
"LIP_STICK_SWING_4",
"LIP_STICK_SWING_DASH",
"ITEM_PARASOL_OPEN",
"ITEM_PARASOL_FALL",
"ITEM_PARASOL_FALL_SPECIAL",
"ITEM_PARASOL_DAMAGE_FALL",
"L_GUN_SHOOT",
"L_GUN_SHOOT_AIR",
"L_GUN_SHOOT_EMPTY",
"L_GUN_SHOOT_AIR_EMPTY",
"FIRE_FLOWER_SHOOT",
"FIRE_FLOWER_SHOOT_AIR",
"ITEM_SCREW",
"ITEM_SCREW_AIR",
"DAMAGE_SCREW",
"DAMAGE_SCREW_AIR",
"ITEM_SCOPE_START",
"ITEM_SCOPE_RAPID",
"ITEM_SCOPE_FIRE",
"ITEM_SCOPE_END",
"ITEM_SCOPE_AIR_START",
"ITEM_SCOPE_AIR_RAPID",
"ITEM_SCOPE_AIR_FIRE",
"ITEM_SCOPE_AIR_END",
"ITEM_SCOPE_START_EMPTY",
"ITEM_SCOPE_RAPID_EMPTY",
"ITEM_SCOPE_FIRE_EMPTY",
"ITEM_SCOPE_END_EMPTY",
"ITEM_SCOPE_AIR_START_EMPTY",
"ITEM_SCOPE_AIR_RAPID_EMPTY",
"ITEM_SCOPE_AIR_FIRE_EMPTY",
"ITEM_SCOPE_AIR_END_EMPTY",
"LIFT_WAIT",
"LIFT_WALK_1",
"LIFT_WALK_2",
"LIFT_TURN",
"GUARD_ON",
"GUARD",
"GUARD_OFF",
"GUARD_SET_OFF",
"GUARD_REFLECT",
"DOWN_BOUND_U",
"DOWN_WAIT_U",
"DOWN_DAMAGE_U",
"DOWN_STAND_U",
"DOWN_ATTACK_U",
"DOWN_FOWARD_U",
"DOWN_BACK_U",
"DOWN_SPOT_U",
"DOWN_BOUND_D",
"DOWN_WAIT_D",
"DOWN_DAMAGE_D",
"DOWN_STAND_D",
"DOWN_ATTACK_D",
"DOWN_FOWARD_D",
"DOWN_BACK_D",
"DOWN_SPOT_D",
"PASSIVE",
"PASSIVE_STAND_F",
"PASSIVE_STAND_B",
"PASSIVE_WALL",
"PASSIVE_WALL_JUMP",
"PASSIVE_CEIL",
"SHIELD_BREAK_FLY",
"SHIELD_BREAK_FALL",
"SHIELD_BREAK_DOWN_U",
"SHIELD_BREAK_DOWN_D",
"SHIELD_BREAK_STAND_U",
"SHIELD_BREAK_STAND_D",
"FURA_FURA",
"CATCH",
"CATCH_PULL",
"CATCH_DASH",
"CATCH_DASH_PULL",
"CATCH_WAIT",
"CATCH_ATTACK",
"CATCH_CUT",
"THROW_F",
"THROW_B",
"THROW_HI",
"THROW_LW",
"CAPTURE_PULLED_HI",
"CAPTURE_WAIT_HI",
"CAPTURE_DAMAGE_HI",
"CAPTURE_PULLED_LW",
"CAPTURE_WAIT_LW",
"CAPTURE_DAMAGE_LW",
"CAPTURE_CUT",
"CAPTURE_JUMP",
"CAPTURE_NECK",
"CAPTURE_FOOT",
"ESCAPE_F",
"ESCAPE_B",
"ESCAPE",
"ESCAPE_AIR",
"REBOUND_STOP",
"REBOUND",
"THROWN_F",
"THROWN_B",
"THROWN_HI",
"THROWN_LW",
"THROWN_LW_WOMEN",
"PASS",
"OTTOTTO",
"OTTOTTO_WAIT",
"FLY_REFLECT_WALL",
"FLY_REFLECT_CEIL",
"STOP_WALL",
"STOP_CEIL",
"MISS_FOOT",
"CLIFF_CATCH",
"CLIFF_WAIT",
"CLIFF_CLIMB_SLOW",
"CLIFF_CLIMB_QUICK",
"CLIFF_ATTACK_SLOW",
"CLIFF_ATTACK_QUICK",
"CLIFF_ESCAPE_SLOW",
"CLIFF_ESCAPE_QUICK",
"CLIFF_JUMP_SLOW_1",
"CLIFF_JUMP_SLOW_2",
"CLIFF_JUMP_QUICK_1",
"CLIFF_JUMP_QUICK_2",
"APPEAL_R",
"APPEAL_L",
"SHOULDERED_WAIT",
"SHOULDERED_WALK_SLOW",
"SHOULDERED_WALK_MIDDLE",
"SHOULDERED_WALK_FAST",
"SHOULDERED_TURN",
"THROWN_F_F",
"THROWN_F_B",
"THROWN_F_HI",
"THROWN_F_LW",
"CAPTURE_CAPTAIN",
"CAPTURE_YOSHI",
"YOSHI_EGG",
"CAPTURE_KOOPA",
"CAPTURE_DAMAGE_KOOPA",
"CAPTURE_WAIT_KOOPA",
"THROWN_KOOPA_F",
"THROWN_KOOPA_B",
"CAPTURE_KOOPA_AIR",
"CAPTURE_DAMAGE_KOOPA_AIR",
"CAPTURE_WAIT_KOOPA_AIR",
"THROWN_KOOPA_AIR_F",
"THROWN_KOOPA_AIR_B",
"CAPTURE_KIRBY",
"CAPTURE_WAIT_KIRBY",
"THROWN_KIRBY_STAR",
"THROWN_COPY_STAR",
"THROWN_KIRBY",
"BARREL_WAIT",
"BURY",
"BURY_WAIT",
"BURY_JUMP",
"DAMAGE_SONG",
"DAMAGE_SONG_WAIT",
"DAMAGE_SONG_RV",
"DAMAGE_BIND",
"CAPTURE_MEWTWO",
"CAPTURE_MEWTWO_AIR",
"THROWN_MEWTWO",
"THROWN_MEWTWO_AIR",
"WARP_STAR_JUMP",
"WARP_STAR_FALL",
"HAMMER_WAIT",
"HAMMER_WALK",
"HAMMER_TURN",
"HAMMER_KNEE_BEND",
"HAMMER_FALL",
"HAMMER_JUMP",
"HAMMER_LANDING",
"KINOKO_GIANT_START",
"KINOKO_GIANT_START_AIR",
"KINOKO_GIANT_END",
"KINOKO_GIANT_END_AIR",
"KINOKO_SMALL_START",
"KINOKO_SMALL_START_AIR",
"KINOKO_SMALL_END",
"KINOKO_SMALL_END_AIR",
"ENTRY",
"ENTRY_START",
"ENTRY_END",
"DAMAGE_ICE",
"DAMAGE_ICE_JUMP",
"CAPTURE_MASTER_HAND",
"CAPTURE_DAMAGE_MASTER_HAND",
"CAPTURE_WAIT_MASTER_HAND",
"THROWN_MASTER_HAND",
"CAPTURE_KIRBY_YOSHI",
"KIRBY_YOSHI_EGG",
"CAPTURE_REDEAD",
"CAPTURE_LIKE_LIKE",
"DOWN_REFLECT",
"CAPTURE_CRAZY_HAND",
"CAPTURE_DAMAGE_CRAZY_HAND",
"CAPTURE_WAIT_CRAZY_HAND",
"THROWN_CRAZY_HAND",
"BARREL_CANNON_WAIT",
"WAIT_1",
"WAIT_2",
"WAIT_3",
"WAIT_4",
"WAIT_ITEM",
"SQUAT_WAIT_1",
"SQUAT_WAIT_2",
"SQUAT_WAIT_ITEM",
"GUARD_DAMAGE",
"ESCAPE_N",
"ATTACK_S_4_HOLD",
"HEAVY_WALK_1",
"HEAVY_WALK_2",
"ITEM_HAMMER_WAIT",
"ITEM_HAMMER_MOVE",
"ITEM_BLIND",
"DAMAGE_ELEC",
"FURA_SLEEP_START",
"FURA_SLEEP_LOOP",
"FURA_SLEEP_END",
"WALL_DAMAGE",
"CLIFF_WAIT_1",
"CLIFF_WAIT_2",
"SLIP_DOWN",
"SLIP",
"SLIP_TURN",
"SLIP_DASH",
"SLIP_WAIT",
"SLIP_STAND",
"SLIP_ATTACK",
"SLIP_ESCAPE_F",
"SLIP_ESCAPE_B",
"APPEAL_S",
"ZITABATA",
"CAPTURE_KOOPA_HIT",
"THROWN_KOOPA_END_F",
"THROWN_KOOPA_END_B",
"CAPTURE_KOOPA_AIR_HIT",
"THROWN_KOOPA_AIR_END_F",
"THROWN_KOOPA_AIR_END_B",
"THROWN_KIRBY_DRINK_S_SHOT",
"THROWN_KIRBY_SPIT_S_SHOT"]
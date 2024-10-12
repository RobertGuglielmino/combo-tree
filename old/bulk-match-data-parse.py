from slippi import Game
import json

testgame = Game('./test user replays/Game_20221028T163330.slp')

# framelist = [frame.Port.leader.pre.state for frame in testgame.frames]



# characters, stage
def getCombos(HERO, VILLAIN, STAGE):
    testgame.start.players[0].character == HERO
    testgame.start.players[1].character == VILLAIN
    testgame.start.stage == STAGE

# compressed states list
count = 0
currState = -1
stateList = []
for frame in testgame.frames:
    stateAtFrame = frame.ports[1].leader.pre.state
    if stateAtFrame == currState:
        count += 1
    else:
        stateList.append(
            {
                "currState": currState,
                "framesAtState": count,
                "absoluteFrame": frame.index - count
            }
        )
        currState = stateAtFrame
        count = 1


print(list(filter(lambda x: x["currState"] == 216, stateList)))

# get ports game.players, tuple

def findFirstCatchFrames(testgame, deathFrames):
    return [list(filter(lambda x: x.currState == 216, testgame[deathFrames[i]: deathFrames[i+1]]))[0].index for i in range(len(deathFrames) - 1)]

def findVillainDeaths(stateList):
    for i in stateList:
        if currState.value < 11:
            print(i.absoluteFrame)
    return [i.absoluteFrame for i in stateList if currState < 11]

# positions, time, stocks, states, direction
def getChainGrabs1(testgame):
    deathFrames = [0] + findVillainDeaths(stateList)
    # print(stateList)
    # print(deathFrames)
    firstCatchFrames = findFirstCatchFrames(testgame, deathFrames)

    # assemble chaingrabs
    

getChainGrabs1(testgame)




for frame in testgame.frames:
    stateAtFrame = frame.ports[1].leader.pre.state












## OLD TEST FOR JUST STATE SEQUENCING

# count = 0
# currState = -1
# stateList = []
# frameCountAtState = []
# for frame in testgame.frames:
#     stateAtFrame = frame.ports[1].leader.pre.state
#     if stateAtFrame == currState:
#         count += 1
#     else:
#         stateList.append(
#             currState
#         )
#         frameCountAtState.append(
#             count
#         )
#         currState = stateAtFrame
#         count = 1

# print([i for i in range(len(stateList)) if stateList[i] == 216])

# print(json.dumps(str(stateList), indent=4))


# Both characters' positions, percents, and action states
# Stage position and platform configuration
# Recent history of inputs (last 30-60 frames)
# Relative positioning between characters
# Current momentum and drift values
    

# framelist = [frame.ports[0] for frame in testgame.frames]
# print(framelist[0].leader.pre)
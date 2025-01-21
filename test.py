commands = [
    "follow the road right, maintain speed", "follow the road right, maintain speed",
    "follow the road right, maintain speed", "follow the road right, maintain speed",
    "follow the road right, maintain speed", "follow the road right, maintain speed",
    "go straight, slow down", "go straight, slow down", "go straight, slow down", 
    "go straight, slow down", "go straight, slow down", "go straight, slow down", 
    "follow the road right, slow down", "follow the road right, slow down", 
    "follow the road right, slow down", "follow the road right, maintain speed", 
    "follow the road right, maintain speed", "follow the road right, maintain speed", 
    "follow the road right, maintain speed", "go straight, maintain speed", 
    "go straight, maintain speed", "go straight, speed up", "go straight, maintain speed", 
    "go straight, maintain speed", "go straight, maintain speed", "go straight, maintain speed", 
    "go straight, maintain speed", "go straight, maintain speed", "go straight, maintain speed", 
    "go straight, slow down", "go straight, maintain speed", "go straight, maintain speed", 
    "go straight, maintain speed"
]


commands_sublist = commands[:27]


result = []
for command in commands_sublist:
    if not result or result[-1] != command:
        result.append(command)

final_command = '. '.join(result)
print(final_command)

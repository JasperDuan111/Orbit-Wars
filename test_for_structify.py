from box import Box, BoxList

b = {'action': BoxList([]), 'info': Box({}), 'observation': Box({'remainingOverageTime': 60, 'step': 0, 'planets': [], 'fleets': [], 'player': 0, 'angular_velocity': 0, 'initial_planets': [], 'next_fleet_id': 0, 'comets': [], 'comet_planet_ids': []}), 'reward': 0, 'status': 'ACTIVE'}
b_schema = {
    'additionalProperties': False,
    'properties':
        Box({'action':
                 {
                    'description': 'List of moves to make. Each move is [from_planet_id, direction_angle, num_ships]',
                    'type': 'array',
                     'default': [],
                     'items': {
                         'type': 'array',
                         'minItems': 3,
                         'maxItems': 3,
                         'items': [
                             {'type': 'integer', 'description': 'Source planet ID'},
                             {'type': 'number', 'description': 'Angle in radians'},
                             {'type': 'integer', 'description': 'Number of ships'}
                         ]
                     }
                 },
            'reward': {'description': 'Score at the end of the game.', 'type': ['number', 'null'], 'default': 0},
            'info': {'description': 'Additional infomation; not available for evaluation.', 'type': 'object', 'default': {}, 'properties': {}},
            'observation': {
                'description': 'Observation to create an action based upon.',
                'type': 'object',
                'additionalProperties': True,
                'properties': {
                    'remainingOverageTime': {'description': 'Total remaining banked time (seconds) that can be used in excess of per-step actTimeouts -- agent is disqualified with TIMEOUT status when this drops below 0.', 'shared': False, 'type': 'number', 'minimum': 0, 'default': 60},
                    'step': {'description': 'Current step within the episode.', 'type': 'integer', 'shared': True, 'minimum': 0, 'default': 0},
                    'planets': {'description': 'List of planets: [id, owner, x, y, radius, ships, production]', 'type': 'array', 'default': [], 'items': {}},
                    'fleets': {'description': 'List of active fleets: [id, owner, x, y, angle, from_planet_id, ships]', 'type': 'array', 'default': [], 'items': {}},
                    'player': {'description': 'Player ID (0, 1, 2, or 3)', 'type': 'integer', 'default': 0},
                    'angular_velocity': {'description': 'Rotation speed of planets in radians per turn.', 'type': 'number', 'default': 0},
                    'initial_planets': {'description': 'Initial planet positions at game start: [id, owner, x, y, radius, ships, production]', 'type': 'array', 'default': [], 'items': {}},
                    'next_fleet_id': {'description': 'Next available fleet ID.', 'type': 'integer', 'default': 0},
                    'comets': {'description': 'Active comet groups with planet_ids, paths, and path_index.', 'type': 'array', 'default': [], 'items': {}},
                    'comet_planet_ids': {'description': 'Planet IDs that are comets (temporary extra-solar objects).', 'type': 'array', 'default': [], 'items': {}}
                },
                'default': {}},
            'status': {
                'description': 'Agent status caused by stepping through the environment.', 'type': 'string', 'default': 'ACTIVE', 'enum': ['INACTIVE', 'ACTIVE', 'DONE', 'ERROR', 'INVALID', 'TIMEOUT']
            }
        }),
    'type': 'object'}

s = {
    'action': [],
    'info': {},
    'observation':
        {'remainingOverageTime': 60,
         'step': 0,
         'planets': [],
         'fleets': [],
         'player': 0,
         'angular_velocity': 0,
         'initial_planets': [],
         'next_fleet_id': 0,
         'comets': [],
         'comet_planet_ids': []
         },
    'reward': 0,
    'status': 'ACTIVE'
}
s_schema = {
    'additionalProperties': False,
    'properties': {
        'action':
            {'description': 'List of moves to make. Each move is [from_planet_id, direction_angle, num_ships]', 'type': 'array', 'default': [], 'items': {}},
        'reward': {'description': 'Score at the end of the game.', 'type': ['number', 'null'], 'default': 0},
        'info': {'description': 'Additional infomation; not available for evaluation.', 'type': 'object', 'default': {}, 'properties': {}}, 'observation': {'description': 'Observation to create an action based upon.', 'type': 'object', 'additionalProperties': True, 'properties': {'remainingOverageTime': {'description': 'Total remaining banked time (seconds) that can be used in excess of per-step actTimeouts -- agent is disqualified with TIMEOUT status when this drops below 0.', 'shared': False, 'type': 'number', 'minimum': 0, 'default': 60}, 'step': {'description': 'Current step within the episode.', 'type': 'integer', 'shared': True, 'minimum': 0, 'default': 0}, 'planets': {'description': 'List of planets: [id, owner, x, y, radius, ships, production]', 'type': 'array', 'default': [], 'items': {}}, 'fleets': {'description': 'List of active fleets: [id, owner, x, y, angle, from_planet_id, ships]', 'type': 'array', 'default': [], 'items': {}}, 'player': {'description': 'Player ID (0, 1, 2, or 3)', 'type': 'integer', 'default': 0}, 'angular_velocity': {'description': 'Rotation speed of planets in radians per turn.', 'type': 'number', 'default': 0}, 'initial_planets': {'description': 'Initial planet positions at game start: [id, owner, x, y, radius, ships, production]', 'type': 'array', 'default': [], 'items': {}}, 'next_fleet_id': {'description': 'Next available fleet ID.', 'type': 'integer', 'default': 0}, 'comets': {'description': 'Active comet groups with planet_ids, paths, and path_index.', 'type': 'array', 'default': [], 'items': {}}, 'comet_planet_ids': {'description': 'Planet IDs that are comets (temporary extra-solar objects).', 'type': 'array', 'default': [], 'items': {}}}, 'default': {}}, 'status': {'description': 'Agent status caused by stepping through the environment.', 'type': 'string', 'default': 'ACTIVE', 'enum': ['INACTIVE', 'ACTIVE', 'DONE', 'ERROR', 'INVALID', 'TIMEOUT']}}, 'type': 'object'}
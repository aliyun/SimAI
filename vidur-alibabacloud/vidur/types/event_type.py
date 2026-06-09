from vidur.types.base_int_enum import BaseIntEnum


class EventType(BaseIntEnum):
    # at any given time step, call the schedule event at the last
    # to ensure that all the requests are processed
    BATCH_STAGE_ARRIVAL = 1
    REQUEST_ARRIVAL = 2
    BATCH_STAGE_END = 3
    BATCH_END = 4
    GLOBAL_SCHEDULE = 5
    REPLICA_SCHEDULE = 6
    REPLICA_STAGE_SCHEDULE = 7

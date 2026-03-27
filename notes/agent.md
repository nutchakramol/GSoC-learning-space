## Agent activation
it can be activated by agents.do("method") or shuffle

**Main idea is:**
    learn to control who act when to act and in which order

**Order**
different of order will give the different output
- diff order will give diff chain reaction
- map func is to report data
- same order every time lead to **bias** , to avoid it use *shuffle_do*

## Scheduling
track time with real simulation time
normal agents.shuffle_do()
    happen **every** time step
    happens every 5 time units
model.schedule_recurring(fn, Schedule(interval=5.0))

model.schedule_event(fn,at=5.0)
    happen **1 time** at specific time

model.schedule_event(fn, after=2.0)
    happens later (relative time)

### Priority
who will do first at same time
    Priority.HIGH
    Priority.DEFAULT
    Priority.LOW





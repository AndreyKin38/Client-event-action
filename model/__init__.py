import dill


def event_action_pipeline():
    with open('model/event_action_pipe.pkl', 'rb') as file:
        model = dill.load(file)

    return model


__all__ = ['event_action_pipeline']


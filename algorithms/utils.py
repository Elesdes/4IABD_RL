from do_not_touch.result_structures import Policy
def return_policy_into_dict(pi) -> Policy:
    returned_policy = {0: dict()}
    for i, pi_i in enumerate(pi):
        returned_policy[0].update({i: pi_i})
    return returned_policy
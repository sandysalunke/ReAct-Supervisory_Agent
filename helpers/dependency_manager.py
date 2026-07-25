# Returns the results for dependencies for current task
def get_dependency_results(task, task_results):
    dependency_results = {}

    for dep in task["dependencies"]:
        dependency_results[dep] = task_results[dep]

    return dependency_results
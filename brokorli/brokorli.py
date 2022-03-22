from .brokorli_unit import BrokorliUnit
from .dashboard import Dashboard
from .workflow import Workflow

class Brokorli:
    
    TASKS = {
        "ner": "named entity recognition",
        "mrc": "machine reading comprehension",
        "qg": "question generation",
        "sm": "sentence and question semantic matching classificaiton"
    }
    
    def __init__(self, additional_templates=None):
        self.tasks = {}
        for task in self.TASKS.keys():
            self.tasks.setdefault(task, BrokorliUnit(task_name=task))

        if additional_templates:
            if type(additional_templates) == list:
                for template in additional_templates:
                    self.tasks["qg"].task.add_templates(template)
            else:
                self.tasks["qg"].task.add_templates(additional_templates)
        
        self.workflow = Workflow(tasks=self.tasks)
        
    def __call__(self, sentence, sm_batch_size=4, mrc_batch_size=4, mrc_threshold=0.9):
        return self.workflow.run(sentence, sm_batch_size=sm_batch_size, mrc_batch_size=mrc_batch_size, mrc_threshold=mrc_threshold)
    
    def available_tasks(self):
        return list(self.TASKS.values())
    
    def run_cli(self):
        self.workflow.cli()
    
    def run_web_demo(self):
        dashboard = Dashboard(tasks=self.tasks)
        dashboard.run()
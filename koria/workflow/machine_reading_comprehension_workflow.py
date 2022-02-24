from . import WorkflowBase


class MrcWorkflow(WorkflowBase):
    
    """
    MRC Workflow
    MRC Workflow는 Question Generation Task와 Machine Reading Comprehension Task를 이용해서 관계형 정보를 추출합니다.
    해당 플로우를 수행하기 위해서는 아래와 같은 Task 들이 필요합니다.
    - Rule-based Question Generation
    - Question-Sentence Matching Classification (예정)
    - Machine Reading Comprehension
    """
    
    def __init__(self):
        super().__init__()
        
    def inference(self, **parameters):
        pass
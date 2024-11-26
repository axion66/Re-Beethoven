import os

class wandbHolder:
    def __init__(self, wandb_store):
        self.log = os.path.join(wandb_store,"wandb_log.log")
        
        
    def log(self, something):
        with (self.log, "w+") as f:
            f.write(something)
            f.write(' ')
    
    
        
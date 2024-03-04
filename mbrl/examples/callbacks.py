import wandb


class WandbCallback:

    def __init__(self, type:str ):

        self.type = type 


    def __call__(self, *inputs):
        if self.type == 'reward':
            obs, action, next_obs, reward, terminated, truncated = inputs
            wandb.log({
            "rollout/ep_rew_mean": reward})
        elif self.type == 'loss':

            model,train_iteration, epoch, total_avg_loss, eval_score, best_val_score = inputs

            wandb.log({
                "epoch": epoch,
                "train_iteration": train_iteration,
                "loss": total_avg_loss,
                "eval_score": eval_score,
                "best_val_score": best_val_score
            })
        else: 
            raise "Unknown logging inputs"
                    
# class WandbCallback:

#     def __init__(self, type:str ):

#         self.type = type 


#     def __call__(self,obs, action, next_obs, reward, terminated, truncated):

#         assert  self.type == 'reward', 'Wrong logging type'

#             # self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
#         wandb.log({
#             "rollout/ep_rew_mean": reward})
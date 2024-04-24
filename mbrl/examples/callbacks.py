import wandb


class WandbCallback:

    def __init__(self, type:str ):

        self.type = type 


    def __call__(self, *inputs):
        if self.type == 'reward':
            obs, action, next_obs, reward, train_ep_reward, step = inputs
            wandb.log({
            "rollout/ep_rew_mean": reward, 
            "rollout/ep_length": step})
    
        elif self.type == 'loss':

            model,train_iteration, epoch, total_avg_loss, eval_score, best_val_score, meta_avg = inputs

            wandb.log({
                "epoch": epoch,
                "train_iteration": train_iteration,
                "loss": total_avg_loss,
                "eval_score": eval_score,
                "best_val_score": best_val_score
            })
            wandb.log({
                "img_loss": meta_avg[0], 
                "kinematic_loss": meta_avg[1], 
                "reward_loss": meta_avg[-1], 
                "kl_loss": meta_avg[-2]
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
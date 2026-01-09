# import torch

# class EarlyStopping:
#     """
#     Early stops training if a validation metric doesn't improve after a given patience.
#     Can handle both minimizing (loss) and maximizing (F1, Accuracy).
#     """

#     def __init__(self, patience=3, delta=0.0, verbose=True, mode='min'):
#         self.patience = patience
#         self.delta = delta
#         self.verbose = verbose
#         self.mode = mode  # 'min' for loss, 'max' for F1/Accuracy
#         self.counter = 0
#         self.early_stop = False
#         self.best_model_state = None
        
#         # Initialize best score based on mode
#         if self.mode == 'min':
#             self.best_score = float('inf')
#         else:
#             self.best_score = float('-inf')

#     def __call__(self, current_score, model):
#         # Determine if the current score is better than the best recorded score
#         if self.mode == 'min':
#             improved = current_score < self.best_score - self.delta
#         else:
#             improved = current_score > self.best_score + self.delta

#         if improved:
#             self.best_score = current_score
#             self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             self.counter = 0
#             if self.verbose:
#                 print(f"Validation {self.mode} score improved to {current_score:.4f}")
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"No improvement ({self.counter}/{self.patience})")
#             if self.counter >= self.patience:
#                 self.early_stop = True
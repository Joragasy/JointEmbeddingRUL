import time
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self,model, train_loader, test_loader, test_label, criterion, optimizer):
        self.model = model
        self.model.to(device)
        self.model = self.model.to(torch.float64)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_label = torch.Tensor(test_label.transpose())
        self.criterion = criterion
        self.optimizer = optimizer
        self.best_model = None
        self.best_score = None
        self.best_eval_loss = 1e9
        self.max_rul = 125
        self.overestimation_penalty = True
        
    def train(self):
        outputs = []
        target = []
        self.model.train()
        Loss = []
        penalty = []
        for i, (x, y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            x = x.to(torch.float64)
            output = self.model(x)
            outputs.append(output)
            target.append(y)
            (loss , p) = self.criterion(self.max_rul*output, self.max_rul*y,overestimation_penalty=self.overestimation_penalty)
            loss.backward()
            self.optimizer.step()
            Loss.append(loss.item())
            penalty.append(p)
        loss_mean = sum(Loss)/len(Loss)
        overestimation_mean = sum(penalty)/len(penalty)
        print(f" --> train loss ( RMSE ) : {loss_mean:.4f} , score : {overestimation_mean:.4f} ")
    
    def evaluate(self,):
        self.model.eval()
        prediction_list = []
        t_label = []
        for j ,(x,y) in enumerate(self.test_loader):
            x = x.to(torch.float64)
            start= time.time()
            prediction = self.model(x)      
            prediction_list.append(prediction)
            t_label.append(y)
        out_batch_pre = torch.cat(prediction_list).detach().cpu().numpy()
        t_label = torch.cat(t_label).detach().cpu()
        prediction_tensor = torch.from_numpy(out_batch_pre)   
        test_loss , overestimation_mean = self.criterion(prediction_tensor*self.max_rul, t_label*self.max_rul,mode="test",overestimation_penalty=self.overestimation_penalty)
        
        print(f'--> test_loss ( RMSE ) = {test_loss.item()} , score : {overestimation_mean:.4f} ')
        
        return test_loss.item()
    
    def save_model(self, file_path, file_name):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        torch.save(self.best_model, file_path + '/' + file_name)
    
    def start_training_loop(self, epochs):
        
        for epoch in range(epochs):
            print('epoch ',epoch,':')
            st = time.time()
            self.train()
            eval_loss = self.evaluate()
            if eval_loss < self.best_eval_loss :
                self.best_score = eval_loss
                self.best_eval_loss = eval_loss
                self.best_model = copy.deepcopy(self.model)
            et = time.time()
            # get the execution time
            elapsed_time = et - st
            print(f"-----> Best loss ( RMSE ) {self.best_score}")
            print('------> Execution time:', elapsed_time, 'seconds')
import torch
from torch.utils.data import Dataset,DataLoader

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every:int
    ) -> None:
        self.gpu_id=gpu_id
        self.model=model.to(gpu_id)
        self.train_dataset=train_dataset
        self.optimizer=optimizer
        self.save_every=save_every
    
    def _run_batch(self,sorce,targets):
        self.optimizer.zero_grads()
        output=self.model(sorce)
        loss = torch.nn.CrossEntropyLoss()(output,targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _run_epoch(self,epoch):
        b_sz = len(next(iter(self.train_dataset))[0]) #how this works? or check if it runs in loop. Last batch will be correctly specized if they are une
        print(f"[GPU{self.gpu_id}] Epoch{epoch} | Batchsize:{b_sz} | Steps:{len(self.train_dataset)}")
        for soruce,target in self.train_dataset:
            source = source.to(self.gpu_id)
            target = target.to(self.gpu_id)
            self._run_batch(soruce,target)
    
    def _save_checkpoint(self,epoch):
        ckp = self.model.state_dict()
        torch.save(ckp, "checkpoint.pt")
        print(f"Epoch{epoch} | Training checkpoint saved at Checkpoint.pt")
        
    def train(self,epochs):
        for epoch in range(epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs():
    train_set = pass
    model = torch.nn.Linear(20,1)
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    return train_set,model,optimizer

def prepare_dataloader(dataset:Dataset,batch_size:int):
    return DataLoader(dataset,batch_size=batch_size,pin_memore= True,shuffle=True)

def main(device,total_epochs, save_every):
    dataset,model,optimzer = load_train_objs()
    train_data = prepare_dataloader(dataset,batch_size=100)
    trainer = Trainer(model,train_data,optimizer,device,save_every)
    trainer.train(total_epochs)

if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    device = 0
    main(device,total_epochs,save_every)
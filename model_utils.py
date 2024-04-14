import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

device = "cuda"

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=4):
        super(UNet, self).__init__()

        features = [64, 128, 256, 512]
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2),
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)
    
def train(model, optimizer, criterion, dataloader, num_epochs):
    global device
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs_mask = model(inputs)

            loss = criterion(outputs_mask, masks.squeeze(1).long())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")
    return running_loss

def validate(model, criterion, dataloader):
    global device
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device)
           
            outputs_mask = model(inputs)
            
            loss = criterion(outputs_mask, masks.squeeze(1))

            running_loss += loss.item()

    print(f"Validation Loss: {running_loss/len(dataloader)}")
    return running_loss

def test(model, dataloader):
    global device
    model.eval()
    total = 0
    loss_score = 0  
    all_inputs = []
    all_masks = []
    all_outputs_mask = []
    all_outputs_softmax = []
    
    criterion = nn.CrossEntropyLoss() 

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device) 
               
            outputs_mask = model(inputs)
            outputs_softmax = torch.softmax(outputs_mask, dim=1)
            predicted = torch.argmax(outputs_softmax, dim=1)

            loss = criterion(outputs_mask, masks.squeeze(1))  
            loss_score += loss.item()
            
            total += masks.size(0)

            all_inputs.append(inputs)
            all_masks.append(masks)
            all_outputs_softmax.append(outputs_softmax)
            all_outputs_mask.append(predicted)

        test_loss = loss_score / total

        print(f"Test Loss is: {test_loss}")
    return test_loss, all_inputs, all_masks, all_outputs_mask, all_outputs_softmax

def train_and_get_validation_loss(model, train_dataset, validation_dataset, criterion, batch_size, learning_rate):
    global num_epochs

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
    _ = train(model, optimizer, criterion, train_dataloader, num_epochs)
    validation_loss = validate(model, criterion, validation_dataloader)
    
    return validation_loss

def store_tested_parameters(res_gp):

    parameters_tested = res_gp.x_iters
    all_scores = res_gp.func_vals

    batch_sizes = [x[0] for x in parameters_tested]
    learning_rates = [x[1] for x in parameters_tested]

    hyperparameter_df = pd.DataFrame(
        {
            "batch_size": batch_sizes,
            "learning_rate": learning_rates,
            "score": all_scores
        }
    )

    hyperparameter_df.to_csv("unet_hyperparameters.csv", index=False)

def train_with_early_stopping(model, optimizer, criterion, dataloader, val_dataloader, num_epochs, patience):
    global device
    best_val_loss = float('inf')  
    early_stop_count = 0  

    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device)
    
            optimizer.zero_grad()
            outputs_mask = model(inputs)
    
            loss = criterion(outputs_mask, masks.squeeze(1).long())
    
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(dataloader)}")
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for data in val_dataloader:
                inputs, masks = data
                inputs, masks = inputs.to(device), masks.to(device)
                
                outputs_mask = model(inputs)
                
                val_loss = criterion(outputs_mask, masks.squeeze(1).long())
                
                val_running_loss += val_loss.item()
        
        avg_val_loss = val_running_loss/len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_count = 0

            torch.save(model.state_dict(), 'model.pt')
        else:
            early_stop_count += 1
            
        if early_stop_count >= patience:
            print("Early stopping!")
            
            model.load_state_dict(torch.load('model.pt'))
            break

        print(f"Early stopping counter: {early_stop_count} out of {patience}")


    return running_loss

def calculate_summary_stats(predicted_masks, actual_masks):
    
    all_preds = []
    all_targets = []

    for batch_preds, batch_masks in zip(predicted_masks, actual_masks):
        all_preds.extend(batch_preds.reshape(-1).cpu().numpy())
        all_targets.extend(batch_masks.reshape(-1).cpu().numpy())

    clf_report = classification_report(all_targets, all_preds)
    
    print(clf_report)

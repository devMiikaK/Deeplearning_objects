from main_script import *

def train_model(model, criterion, optimizer, trainloader, epochs=5):
    training_losses = []
    for e in range(epochs):
        atm_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            logits = model(images)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            atm_loss += loss.item()

        e_loss = atm_loss / len(trainloader)
        training_losses.append(e_loss)
        print(f"Training loss: {e_loss:.6f}")
    return training_losses
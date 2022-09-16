import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
from torchvision import datasets, transforms

if __name__ == "__main__":
    quit()
    print('2 FC Layers Neural Network')

    # Device
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print('CUDA available. Training on GPU.') if cuda_available else print(
        'Training on CPU.')

    # Hyper Parameters
    input_size = 784
    hidden_size = 500
    num_classes = 10
    batch_size = 100
    num_epochs = 5
    learning_rate = 0.001
    MNIST_data_path = '/home/jerome/data/'

    # MNIST Dataset
    train_dataset = datasets.MNIST(MNIST_data_path,
                                   train=True,
                                   download=False,
                                   transform=transforms.Compose([transforms.ToTensor(),
                                                                transforms.Normalize(
                                       mean=(0.1307,), std=(0.3081,))])
                                   )

    test_dataset = datasets.MNIST(MNIST_data_path,
                                  train=False,
                                  download=False,
                                  transform=transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize(
                                      mean=(0.1307,), std=(0.3081,))])
                                  )

    # Number of samples in the dataset
    num_train_samples = len(train_dataset)
    num_test_samples = len(test_dataset)

    # Data Loaders
    train_loader = tud.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = tud.DataLoader(test_dataset, batch_size=batch_size)

    print('Data Loaded!')

    # Neural Network Model
    model = FCnet(input_size, hidden_size, num_classes)
    model.to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print('Start Training...')
    for epoch in range(num_epochs):
        # Initialize
        running_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            batch_num = data.size(0)
            data, target = data.view(
                batch_num, -1).to(device), target.to(device)

            # Forward Pass
            output = model(data)
            loss = loss_fn(output, target)

            # Update running loss
            running_loss += loss.item() * batch_num

            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        sample_mean_loss = running_loss / num_train_samples
        print(
            f'Epoch [{epoch + 1}/{num_epochs}]  Training Loss: {sample_mean_loss}')
    print('Start Testing...')
    model.eval()
    num_correct = 0
    for i, (data, target) in enumerate(test_loader):
        batch_num = data.size(0)
        data, target = data.view(batch_num, -1).to(device), target.to(device)
        output = model(data)
        prediction = output.argmax(dim=1)
        num_correct += torch.sum(prediction.eq(target))
    test_accuracy = num_correct / num_test_samples
    print(f'Testing Accuracy: {test_accuracy}')

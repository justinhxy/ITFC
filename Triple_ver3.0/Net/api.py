import sys
import torch
import torch.utils.data
from tqdm import tqdm

def run(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for i, (img, label) in enumerate(train_bar):
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (img, label) in enumerate(test_bar):
                img = img.to(device)
                label = label.to(device)
                outputs = model(img)
                _, predictions = torch.max(outputs, dim=1)
                observer.update(predictions, label)
        observer.excute(epoch)
    observer.finish()

def run_main_1(observer, epochs, train_loader, test_loader, model, device, optimizer, criterion):
    model = model.to(device)
    print("start training")
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        observer.reset()
        model.train()
        train_bar = tqdm(train_loader, leave=True, file=sys.stdout)

        for ii, (mri_images, pet_image, cli_tab, label) in enumerate(tqdm(train_bar)):
            if torch.isnan(mri_images).any():
                print("train: NaN detected in input mri_images")
            if torch.isnan(pet_image).any():
                print("train: NaN detected in input pet_image")
            mri_images = mri_images.cuda(non_blocking=True)
            pet_image = pet_image.cuda(non_blocking=True)
            cli_tab = cli_tab.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            optimizer.zero_grad()
            mri_feature, pet_feature, cli_feature, outputs = model.forward(mri_images, pet_image, cli_tab)
            # print(f'feature before loss{mri_feature.shape}')
            loss = criterion(mri_feature, pet_feature, cli_feature, label, outputs)
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            model.eval()
            test_bar = tqdm(test_loader, leave=True, file=sys.stdout)

            for i, (mri_images, pet_image, cli_tab, label) in enumerate(test_bar):
                mri_images = mri_images.cuda(non_blocking=True)
                pet_image = pet_image.cuda(non_blocking=True)
                cli_tab = cli_tab.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                _, _, _, outputs = model.forward(mri_images, pet_image, cli_tab)
                _, predictions = torch.max(outputs, dim=1)
                observer.update(predictions, label)

        if observer.excute(epoch, model=model):
            print("Early stopping")
            break
    observer.finish()
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from dcGAN_image_generation import logger
from dcGAN_image_generation.entity.config_entity import ModelTrainingConfig

 
 


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")


    # -------------------------------
    # DataLoader
    # -------------------------------
    def get_dataloader(self):

        transform = transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.CenterCrop(self.config.image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

        dataset = datasets.ImageFolder(
            root=self.config.data_path / "train",
            transform=transform
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )

        logger.info(f"Dataset loaded with {len(dataset)} images")

    # -------------------------------
    # Weight initialization
    # -------------------------------
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # -------------------------------
    # Generator
    # -------------------------------
    def build_generator(self):

        nz = self.config.z_dim
        ngf = self.config.generator_feature_maps
        nc = 3

        self.netG = nn.Sequential(

            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        ).to(self.device)

        self.netG.apply(self.weights_init)

    # -------------------------------
    # Discriminator
    # -------------------------------
    def build_discriminator(self):

        ndf = self.config.discriminator_feature_maps
        nc = 3

        self.netD = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ).to(self.device)

        self.netD.apply(self.weights_init)

    # -------------------------------
    # Training
    # -------------------------------
    def train(self):

        criterion = nn.BCELoss()

        fixed_noise = torch.randn(64, self.config.z_dim, 1, 1, device=self.device)

        real_label = 1.
        fake_label = 0.

        optimizerD = optim.Adam(
            self.netD.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, 0.999)
        )

        optimizerG = optim.Adam(
            self.netG.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, 0.999)
        )

        logger.info("Starting Training Loop...")

        for epoch in range(self.config.num_epochs):
            for i, data in enumerate(self.dataloader):

                ############################
                # Train Discriminator
                ############################
                self.netD.zero_grad()

                real_images = data[0].to(self.device)
                b_size = real_images.size(0)

                label = torch.full((b_size,), real_label, device=self.device)

                output = self.netD(real_images).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()

                noise = torch.randn(b_size, self.config.z_dim, 1, 1, device=self.device)
                fake = self.netG(noise)

                label.fill_(fake_label)

                output = self.netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()

                optimizerD.step()

                ############################
                # Train Generator
                ############################
                self.netG.zero_grad()

                label.fill_(real_label)

                output = self.netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()

                optimizerG.step()

                if i % self.config.save_image_interval == 0:
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()

                    utils.save_image(
                        fake,
                        self.config.generated_images_dir / f"epoch_{epoch}_iter_{i}.png",
                        normalize=True
                    )

            logger.info(f"Epoch [{epoch+1}/{self.config.num_epochs}] completed")

        torch.save(
            self.netG.state_dict(),
            Path(self.config.trained_model_dir) / "generator.pth"
        )

        torch.save(
            self.netD.state_dict(),
            Path(self.config.trained_model_dir) / "discriminator.pth"
        )

        logger.info("Models saved successfully")
        
        return errG.item(), (errD_real + errD_fake).item()
        
    
   


## Please note due to Local Laptop memory constrained and Since its not explicitly 
## asked I have skipped the Saving the model Output steps to MySQL code in this
## assignment of MLOps flow

________________________________________
PHASE 0: Local Deployment
________________________________________
cd /Users/abhinavpaul/Desktop/Goals/Exec Edu/1. ISB AMPBA/Course Material/Term 2/5. CT1/Assignment/mlops_bank_project
docker compose up --build
# this will run the docker-compose.yaml file where we have defined the bank prediction app
________________________________________
PHASE 1: AWS CONSOLE SETUP
________________________________________
Step 1: Create EC2 Instance (Ubuntu)
1.Go to EC2 > Launch Instance (named - Abhinav_AMPBA_T2)
2.Choose Ubuntu 22.04 LTS as AMI
3.Select instance type: free tier eligible (t2.micro or t3.micro)
4.Create new key pair (named - Abhinav_AMPBA_T2)
5.Allocate storage
6.Create a new Security Group with these inbound rules:
   SSH (port 22) from My IP
   HTTP (port 80) from 0.0.0.0/0
   Custom TCP (port 8501) from 0.0.0.0/0 (for Streamlit app)
   MySQL/Aurora (port 3306) from 0.0.0.0/0 or your IP (for MySQL test access)
7.Launch instance
________________________________________
Step 2: Create RDS MySQL Database
1.Go to RDS > Create Database
2.Choose MySQL
2.Choose free tier
4.Set DB Name: database-1
5.Username: <****Username not sharing here****>, Password: <****password not sharing here****>
6.Enable public access
________________________________________
Step 3: Modify RDS Security Group to Accept EC2 Traffic
1.On RDS instance page locate > vpc Security Group 
2.Select the security group attached to RDS
3.Edit Inbound Rules
4.Add:
   Type: MySQL/Aurora
   Port: 3306
   Source: 0.0.0.0/0 (internet)
________________________________________
Step 4: Create ECR Repository
1.Go to ECR > Create Repository
2.Name: abhinav_ampba_t2
3.Set visibility as private
4.Note the repository URI
________________________________________
Step 5: IAM Setup
IAM User for Local Machine:
1.Go to IAM > Users > Add User
2.Set name abhinav_ampba_t2
3.Attach policy: AdministratorAccess
4.click on user > security credentials > create access keys and download acceskeys.csv (****Not sharing the Keys detail as submission****)
IAM Role for EC2:
1.Go to IAM > Roles > Create Role
2.Use Case: EC2
3.Attach policy: AmazonEC2ContainerRegistryFullAccess
4.Name the role: abhinav_ampba_t2
5.Attach this role to your EC2 instance under Actions > Security > Modify IAM Role
________________________________________
PHASE 2: LOCAL MACHINE SETUP
________________________________________
Step 6: Install and Configure AWS CLI
1.Download and install AWS CLI v2
2.Open terminal and verify installation:
aws --version
3.Configure CLI:
aws configure
	Enter Access Key ID from IAM user (****Not sharing the Keys detail as submission****)
	Enter Secret Access Key (****Not sharing the Keys detail as submission****)
	Region: ap-southeast-2
	Output format: None
________________________________________
Step 7: Navigate to Project and Push Docker Image
1.Open terminal and move to your project folder:
cd /Users/abhinavpaul/Desktop/Goals/Exec Edu/1. ISB AMPBA/Course Material/Term 2/5. CT1/Assignment/mlops_bank_project # path in local
This folder includes:
	Dockerfile
   docker-compose.yaml (with app servic details)
	requirements.txt
	api.py (Streamlit app script) 
   best_model.joblib (pipeline file)
   best_fit_model.pkl (model file)
	data/ban-additionl.csv (training dataset file)
   data/validattion.csv (validation dataset file created using rows from the training data set with few changes used for testing the model)

2.Authenticate Docker to ECR (you can get these commands(Authenticate,build,push) from AWS ECR console --> view push commands):
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin <*****not sharing the details in submission*****>
3.Build and tag the image:
docker build -t abhinav_ampba_t2:latest .
docker tag abhinav_ampba_t2:latest <*****not sharing the details in submission*****>
4.Push image to ECR:
docker push <*****not sharing the details in submission*****>
________________________________________
PHASE 3: EC2 SETUP AND DEPLOYMENT
________________________________________
Connect the EC2 Instance and run below codes
________________________________________
Step 8: Install Docker and AWS CLI

sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker ubuntu
newgrp docker
Install AWS CLI:

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
Verify:

docker --version
aws --version
________________________________________
Step 9: Authenticate Docker to ECR (IAM Role Auto Used) -(get this command from aws ECR--> view push commands)

aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin <*****not sharing the details in submission*****>
________________________________________
Step 10: Pull Docker Image and Inspect
1.Pull the image:
docker pull <*****not sharing the details in submission*****>

2.View image:
docker images
________________________________________
Step 11: Run Docker Container

docker rm bank_prediction_app ##to remove the previous container installed
docker run -d \
  --name bank_prediction_app \
  -e RDS_ENDPOINT=<*****not sharing the details in submission*****> \
  -p 8501:8501 \
  <*****not sharing the details in submission*****>

Open in browser:

http://<*****not sharing the details in submission*****>:8501


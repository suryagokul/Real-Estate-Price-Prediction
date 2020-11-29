# Real-Estate-Price-Prediction

AWS EC2 INSTANCE DEPLOYMENT STEPS ------->

	1) Flask Code.


	2) Run and Check in Local.


	3) Create AWS account  (Doesn't takes rupay card but it takes visa,mastercard...)


		If debit card is rupay then we have to create virtual visa debit card.
				
			1) To do that goto irctcimudra.com signup and Login (8500003899). 

			2) Add money to imudra wallet using rupay card. AWS takes 2 RS from our account.So,we need atleast 2RS in wallet.

			3)Then it creates a visa card. To check the card details click "View Cards" in imudra portal.

			4)Enter this debit card details into aws account.

	4)Create EC2 Instance.


	5)Download Putty and Putyygen.


	6)Load pem file created in ec2 into puttygen and Generate ppk (private key) file. 


	7)Download WinSCP.
	

	8)Update the host and port in app.py as app.run(host='0.0.0.0', port=8080).

	
	9)Install the libraries by using putty which are in requirements.txt.

	
	10)Run python3 app.py in putty.

	
	11)Copy Public ipv4 DNS and paste it in url along with :8080.
		
		EX:  http://ec2-3-12-165-91.us-east-2.compute.amazonaws.com:8080/ Final URL 

	

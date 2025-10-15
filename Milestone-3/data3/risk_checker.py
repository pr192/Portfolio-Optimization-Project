import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_alert(rule_name, portfolio_id):
    sender_email = "myalerts@gmail.com"          
    sender_password = "your_app_password_here"
    receiver_email = "client_email@example.com"  

    subject = f"Risk Alert: {rule_name} failed"
    body = f"The rule '{rule_name}' failed for Portfolio ID: {portfolio_id}."

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Connect to Gmail’s SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"✅ Email sent to {receiver_email}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

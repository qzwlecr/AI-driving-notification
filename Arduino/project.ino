const int motor1 =  2;      
const int motor2 =  3; 
const int motor3 =  4; 
int oldv;
int newv;
int t;

void setup() 
{
  pinMode(motor1, OUTPUT); 
  pinMode(motor2, OUTPUT);
  pinMode(motor3, OUTPUT);  
  Serial.begin(9600); 
}

void loop()
{
  char var;
  if(Serial.available()>0)
  {
    oldv=Serial.read();
    newv=Serial.read();
    if(oldv=newv)
    {
     
    if(newv=='0')
    {
      Serial.print(newv);
       for(t=0;t<5;t++)
      {
        digitalWrite(motor1,HIGH);
        delay(50);
        digitalWrite(motor1,LOW);
        delay(50);
       }
      delay(500);
    }
    if(newv=='1')
    {
      Serial.print(var);
       for(t=0;t<5;t++)
      {
        digitalWrite(motor2,HIGH);
        delay(50);
        digitalWrite(motor2,LOW);
        delay(50);
       }
      delay(500);
    }
    if(newv=='2')
    {
      Serial.print(newv);
       for(t=0;t<5;t++)
      {
        digitalWrite(motor3,HIGH);
        delay(50);
        digitalWrite(motor3,LOW);
        delay(50);
       }
      delay(500);
    }
  }
  }
}

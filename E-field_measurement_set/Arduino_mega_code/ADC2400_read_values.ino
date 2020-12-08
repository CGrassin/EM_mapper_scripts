/* LTC2400 24 Bit ADC Test
* Connect an LTC2400 24 Bit ADC to the Arduino Board in SPI Mode
*
*
*
* KHM 2009 /  Martin Nawrath
* Kunsthochschule fuer Medien Koeln
* Academy of Media Arts Cologne
* 
* Modifyed by Egor Kretov 2020
* for E-field probe value reading

*/
#include <Stdio.h>

#ifndef cbi
#define cbi(sfr, bit)     (_SFR_BYTE(sfr) &= ~_BV(bit))
#endif
#ifndef sbi
#define sbi(sfr, bit)     (_SFR_BYTE(sfr) |= _BV(bit))
#endif

#define LTC_CS 0         // LTC2400 Chip Select Pin  on Portb 0
#define LTC_CS2 0         // LTC2400 Chip2 Select Pin  on PortL 0
#define LTC_MISO  3      // LTC2400 SDO Select Pin  on Portb 3
#define LTC_SCK  1       // LTC2400 SCK Select Pin  on Portb 1

bool ChipSel =0;

void setup() {

 cbi(PORTB,LTC_SCK);      // LTC2400 SCK low
 sbi (DDRB,LTC_CS);       // LTC2400 CS HIGH
 sbi (DDRL,LTC_CS2);       // LTC2400 CS HIGH

 cbi (DDRB,LTC_MISO);
 sbi (DDRB,LTC_SCK);

 Serial.begin(57600);
 // init SPI Hardware
 sbi(SPCR,MSTR) ; // SPI master mode
 sbi(SPCR,SPR0) ; // SPI speed
 sbi(SPCR,SPR1);  // SPI speed
 sbi(SPCR,SPE);   //SPI enable

 Serial.println("LTC2400 ADC Test");

}
float volt;
float v_ref=4.096;          // Reference Voltage, 5.0 Volt for LT1021 or 3.0 for LP2950-3

long int ltw = 0;         // ADC Data ling int
int cnt;                  // counter
byte b0;                  //
byte sig;                 // sign bit flag
char st1[20];             // float voltage text

/********************************************************************/
void loop() {
 if(ChipSel){
  cbi(PORTL,LTC_CS2);
 } // LTC2400 CS2 low
 else {
 cbi(PORTB,LTC_CS);
 }            // LTC2400 CS Low

 
 delayMicroseconds(1);
 if (!(PINB & (1 << PB4))) {    // ADC Converter ready ?
   //    cli();
   ltw=0;
   sig=0;

   b0 = SPI_read();             // read 4 bytes adc raw data with SPI
   if ((b0 & 0x20) ==0) sig=1;  // is input negative ?
   b0 &=0x1F;                   // discard bit 25..31
   ltw |= b0;
   ltw <<= 8;
   b0 = SPI_read();
   ltw |= b0;
   ltw <<= 8;
   b0 = SPI_read();
   ltw |= b0;
   ltw <<= 8;
   b0 = SPI_read();
   ltw |= b0;

   delayMicroseconds(1);

   sbi(PORTB,LTC_CS);           // LTC2400 CS Low
   

  // if (sig) ltw |= 0xf0000000;    // if input negative insert sign bit
   ltw=ltw/16;                    // scale result down , last 4 bits have no information
   volt = ltw * v_ref / 16777216; // max scale

  // Serial.print(cnt++);
  // Serial.print(";  ");
   printFloat(volt,6);           // print voltage as floating number
   Serial.println("  ");
   ChipSel = !ChipSel;
   delay(60);
 }

if(!ChipSel){
  sbi(PORTL,LTC_CS2);
  } // LTC2400 CS2 low
else {
  sbi(PORTB,LTC_CS);
  }            // LTC2400 CS Low
 delay(10);

}
/********************************************************************/
byte SPI_read()
{
 SPDR = 0;
 while (!(SPSR & (1 << SPIF))) ; /* Wait for SPI shift out done */
 return SPDR;
}
/********************************************************************/
//  printFloat from  tim / Arduino: Playground
// printFloat prints out the float 'value' rounded to 'places' places
//after the decimal point
void printFloat(float value, int places) {
 // this is used to cast digits
 int digit;
 float tens = 0.1;
 int tenscount = 0;
 int i;
 float tempfloat = value;

 // if value is negative, set tempfloat to the abs value

   // make sure we round properly. this could use pow from
 //<math.h>, but doesn't seem worth the import
 // if this rounding step isn't here, the value  54.321 prints as

 // calculate rounding term d:   0.5/pow(10,places)
 float d = 0.5;
 if (value < 0)
   d *= -1.0;
 // divide by ten for each decimal place
 for (i = 0; i < places; i++)
   d/= 10.0;
 // this small addition, combined with truncation will round our

 tempfloat +=  d;

 if (value < 0)
   tempfloat *= -1.0;
 while ((tens * 10.0) <= tempfloat) {
   tens *= 10.0;
   tenscount += 1;
 }


 if (tenscount == 0)
   Serial.print(0, DEC);

 for (i=0; i< tenscount; i++) {
   digit = (int) (tempfloat/tens);
   Serial.print(digit, DEC);
   tempfloat = tempfloat - ((float)digit * tens);
   tens /= 10.0;
 }

 // if no places after decimal, stop now and return
 if (places <= 0)
   return;

 // otherwise, write the point and continue on
 Serial.print('.');

 for (i = 0; i < places; i++) {
   tempfloat *= 10.0;
   digit = (int) tempfloat;
   Serial.print(digit,DEC);
   // once written, subtract off that digit
   tempfloat = tempfloat - (float) digit;
 }
}

# LLM-MT-Eval

This repo evaluates

* DeepL
* Google Trans
* WMT22 Best
* text-davinci-003
* gpt-3.5-turbo-0301
* gpt-4-0314

in automatic metrics:

* COMET
* BLEURT
* BLEU
* chrF
* chrF++

 on WMT22 general translation tasks:

* English<->German
* English<->Czech
* English<->Russian
* English<->Chinese
* German<->French
* English<->Japanese
* Ukrainian<->English
* Ukrainian<->Czech
* English->Croatian.



### Results

#### Full results

<p align="center">
<img src="imgs/main.png" alt="main", width="600"/>
</p>



#### **Average performance**

**All language pairs (except for English->Croatian)**

<p align="center">
<img src="imgs/avg_all.png" alt="avg_all", width="400"/>
</p>

**High resource**

* En<->De, En<->Cs, En<->Ru, En<->Zh

<p align="center">
<img src="imgs/avg_high.png" alt="avg_high", width="400"/>
</p>

**Medium resource**

* De<->Fr, En<->Ja, En<->Uk

<p align="center">
<img src="imgs/avg_high.png" alt="avg_mid", width="400"/>
</p>

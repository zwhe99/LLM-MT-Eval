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
* English->Croatian

and WMT21 news translation tasks:

* English<->Hausa
* English<->Icelandic



### Results

#### System outputs

```
output/
|-- deepl
|-- google-cloud
|-- gpt-3.5-turbo-0301
|-- gpt-4-0314
|-- text-davinci-003
`-- wmt-winner
```

#### Full results

<p align="center">
<img src="imgs/main.png" alt="main", width="600"/>
</p>



#### **Average performance**

**All language pairs** （except for those not supported by DeepL）

<p align="center">
<img src="imgs/avg_all.png" alt="avg_all", width="500"/>
</p>

**High resource**

* En<->De, En<->Cs, En<->Ru, En<->Zh

<p align="center">
<img src="imgs/avg_high.png" alt="avg_high", width="500"/>
</p>

**Medium resource**

* De<->Fr, En<->Uk, En<->Ja

<p align="center">
<img src="imgs/avg_mid.png" alt="avg_mid", width="500"/>
</p>

**Low resource**

* Uk<->Cs, En<->Hr, En<->Ha, En<->Is

<p align="center">
<img src="imgs/avg_low.png" alt="avg_low", width="500"/>
</p>

### Evaluation

```sh
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
python3 evaluation/eval.log --bleurt-ckpt BLEURT-20
```

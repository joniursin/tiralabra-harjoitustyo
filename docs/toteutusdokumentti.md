Ohjelman yleisrakenne on neuroverkko, joka luodaan tiedostossa 
```bash
network.py
```
Ohjelma luo 3 kerroksisen neuroverkon annetuilla tasoilla ja lisää sille painot ja taipumukset.
Tämän jälkeen käytetään MNIST tietokannan 70000 kuvaa treenaamaan neuroverkon vastavirtaalgoritmiä käyttäen.
Seuraavaksi testataan neuroverkon tarkkuus syöttämällä sille 10000 kuvaa joiden arvoa verkko ei tiedä ja tästä lasketaan verkon tarkkuus.

Parannusehodotuksina työhön olisi esimerkiksi mini-batchien käyttö verkon treenaamiseen, joka parantaisi tarkkuutta, sekä useamman piilokerroksen käyttö neuroverkossa.

ChatGPT, käytin poetry riippuvuuksien versioiden muuttamiseen yhteensopiviksi. Koodia en ole generoinut ChatGPT:n avulla.

Viitteet:
http://neuralnetworksanddeeplearning.com/chap1.html
https://theneuralblog.com/forward-pass-backpropagation-example/

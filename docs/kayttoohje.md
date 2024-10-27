Ohjelmaa voidaan käyttää kloonaamalla ensin tämä repositio:
```bash
git clone https://github.com/joniursin/tiralabra-harjoitustyo
```
 
Tämän jälkeen asenna riippuvuudet ajamalla
```bash
poetry install
```

Tämän jälkeen siirry shelliin ajamalla
```bash
poetry shell
```
Ohjelma ajetaan komennolla
```bash
python3 canvas.py
```

Ohjelma luo ja treenaa neuroverkon käyttäen MNIST dataa käsinpiirretyistä numeroista. 
Treenauksen jälkeen voidaan testataan sen tarkkuutta piirtämällä omia numeroita aukeavaan piirtoruutuun, joista ohjelma tulostaa piirtoruutuun arvion numerosta prosentteina.

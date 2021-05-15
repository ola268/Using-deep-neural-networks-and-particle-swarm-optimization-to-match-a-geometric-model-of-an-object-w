## Wykorzystanie optymalizacji metodą roju cząstek oraz głębokich sieci neuronowych do dopasowania modelu geometrycznego obiektu do jego zdjęcia

W celu dopasowania modelu geometrycznego wykorzytano implmenetację algorytmu PSO zaczęrpniętą z https://github.com/Witek-/PSOforOPENCV

W celu segmentacji instancji pojazdów wykorzystano implemnetację algorytmu neuronowego YOLACT++ zaczęrpniętą z https://github.com/dbolya/yolact

## Użycie

Zaimplementowane rozwiązanie dopasowuje 2 modele geometryczne samochodu (prosty w postaci prostopadłościaniu oraz złożony, który kształtem przypomina pojazd) do jego maski uzyskanej z procesu segmentacji. W rezultacie otrzymujemy podstawowe informacje o położeniu pojazdu w przestrzeni trójwymiarowej oraz jego przybliżone wymiary. 


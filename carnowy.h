#pragma once
#include "opencv2\opencv.hpp"

using namespace cv;
using namespace std;


class new_car
{
private:
	// wspó³rzedne œrodka, wymiary i kat
	float x, y, z, //to wiadomo
		lx, ly, lz, //d³ugoœci pe³ne w osi x,y,z
		//poni¿ej wspó³czynnniki okreœlaj¹ce
		wf, //szyba przód
		m, //maska
		wb,//szyba ty³
		wr, //szerokoœæ dachu w y
		hf, //wysokoœæ przód
		hb,//wysokoœæ ty³
		r, //dach po x
		angle;
	//float h = float(2.0 / 3.0);
	//macierze z modelem obiektu; model podstawowy, bie¿¹cy (po transformacji) i po rzutowaniu na p³aszczyznê obrazu
	Mat base_model3D, model3D, model2D;
	//macierz przekszta³cenia modelu podstawowego do bie¿¹cego
	Mat T;

	//tworzy model bazowy na podstawie parametrów oraz przekszta³ca go do modelu bie¿¹cego
	void create_model()
	{
		//wygodnie kolejne punkty podawaæ wierszami
		//od razu wspó³rzêdne uogólnione
		//model podstawowy, œrodek podstawy w punkcie 0,0,0
		//wf wiêc punkty po pó³ d³ugoœci w ka¿dym wymiarze na minusie i plusie


		base_model3D = (Mat_<float>(16, 4) <<
			//podstawa
			-lx * .5, -ly * .5, 0, 1,
			lx * .5, -ly * .5, 0, 1,
			lx * .5, ly * .5, 0, 1,
			-lx * .5, ly * .5, 0, 1,

			//PUNKTY PARAMI ID¥C OD PRZODU

			//przód
			lx * .5, -ly * .5, lz * hf, 1,
			lx * .5, ly * .5, lz * hf, 1,
			//pocz¹tek szyby
			lx * .5 - m * lx, -ly * .5, lz * hb, 1,
			lx * .5 - m * lx, ly * .5, lz * hb, 1,
			//koniec szyby
			lx * .5 - m * lx - wf * lx, -ly * .5 * wr, lz, 1,
			lx * .5 - m * lx - wf * lx, ly * .5 * wr, lz, 1,
			//pocz¹tek szyby tylnej czyli te¿ koniec góry 
			lx * .5 - m * lx - wf * lx - r * lx, -ly * .5 * wr, lz, 1,
			lx * .5 - m * lx - wf * lx - r * lx, ly * .5 * wr, lz, 1,
			//koniec szyby tylnej
			lx * .5 - m * lx - wf * lx - r * lx - wb * lx, -ly * .5 , lz*hb, 1,
			lx * .5 - m * lx - wf * lx - r * lx - wb * lx, ly * .5,   lz*hb, 1,
			//ty³
			-lx * .5, -ly * .5, lz * hb, 1,
			-lx * .5, ly * .5, lz * hb, 1);


		//transpozycja modelu, by wspó³rzêdne by³y pionowo - wygodniej do póŸniejszych obliczeñ
		base_model3D = base_model3D.t();

		//wyznaczenie k¹ta, cosinusa i sinusa - liczymy to raz zamiast kilka razy - szybciej
		float a = angle * 0.0174532925;
		float c = cos(a);
		float s = sin(a);

		//obrót wokó³ osi z i przesuniecie - jedna macierz transformacji
		T = (Mat_<float>(4, 4) <<
			c, -s, 0, x,
			s, c, 0, y,
			0, 0, 1, z,
			0, 0, 0, 1);

		//przekszta³cenie modelu bazowego do bie¿¹cego - wspó³rzêdne ju¿ s¹ jednorodne, wiêc po prostu mno¿enie przez T
		model3D = T * base_model3D;
	}
	//powy¿sze zmienne oraz metoda create s¹ prywatne - nie mamy do nich dostêpu z zewn¹trz
	//aby czegoœ przypadkiem nie popsuæ, np. zmienimy x i myœlimy, ¿e ca³y model siê zmieni³.
	//wf tak nie jest

public:
	//konstruktor - przyjmuje domyœlne parametry prostopad³oœcianu
	new_car(float lx = 10, float ly = 10, float lz = 10, float x = 0, float y = 0, float m = 1, float wf = 1, float r = 1, float wb = 1, float wr = 10, float hf = 7, float hb = 7, float angle = 0) {

		this->x = x;
		this->y = y;
		this->angle = angle;
		this->lx = lx;
		this->ly = ly;
		this->lz = lz;
		this->wf = wf;
		this->m = m;
		this->wb = wb;
		this->wr = wr;
		this->r = r;
		this->hf = hf;
		this->hb = hb;

		create_model();
	}

	//zmienia parametry modelu
	//metoda nie jest optymalna, bo za ka¿dym razem tworzy model od nowa, podczas gdy w przypadku przesuniêcia po osi x
	//wystarczy zmodyfikowaæ jedn¹ pozycjê w macierzy T i wykonaæ mno¿enie istniej¹cego modelu
	//jednak algorytm PSO i tak zmienia najczêœciej wszystkie parametry modelu, wiêc tak jest doœæ wygodnie
	//choæ szybciej by³oby zaktualizowaæ


	//rzutowanie modelu bie¿¹cego 3D na p³aszczyznê obrazu (do: model2D)
	void project(Mat& rot, Mat& trans, Mat& cam, Mat& dist)
	{
		//rzutowanie musi byæ we wspó³rzêdnych "normalnych", nie jednorodnych
		//wiêc obcinamy w locie ostatni wiersz (od 0 do 3 WY³¹cznie, czyli 0, 1, 2)
		projectPoints(model3D.rowRange(0, 3), rot, trans, cam, dist, model2D); //model2D to dwukana³owa macierz wierszowa
		model2D = model2D.reshape(1).t(); //przekszta³camy do jednokana³owej macierzy kolumnowej
		//mog³aby byæ i wierszowa, wszystko jedno, ale ¿eby siê nie myli³o z modelem3D, który jest kolumnowy
	}

	//rysuje model2D na obrazie
	void draw(Mat& img, int xs = 0, int ys = 0)
	{
		//metoda zak³ada, ¿e model2D istnieje - jak bêdzie pusty, wyrzuci b³¹d
		//oczywiœcie uk³ad œcian jest dla prostopad³oœcianu - dla innego kszta³tu trzeba zmodyfikowaæ
		vector<Point> face;
		vector<vector<Point>> faces;

		//sciana pozioma dolna
		/*face.push_back((Point)model2D.col(0) + Point(xs,ys));
		face.push_back((Point)model2D.col(1) + Point(xs,ys));
		face.push_back((Point)model2D.col(2) + Point(xs,ys));
		face.push_back((Point)model2D.col(3) + Point(xs,ys));
		faces.push_back(face);*/

		//sciana pozioma gora
		face.clear();
		face.push_back((Point)model2D.col(8) + Point(xs,ys));
		face.push_back((Point)model2D.col(9) + Point(xs,ys));
		face.push_back((Point)model2D.col(11) + Point(xs,ys));
		face.push_back((Point)model2D.col(10) + Point(xs,ys));
		faces.push_back(face);

		//sciana pozioma maska przod
		face.clear();
		face.push_back((Point)model2D.col(4) + Point(xs,ys));
		face.push_back((Point)model2D.col(5) + Point(xs,ys));
		face.push_back((Point)model2D.col(7) + Point(xs,ys));
		face.push_back((Point)model2D.col(6) + Point(xs,ys));
		faces.push_back(face);

		//sciana pozioma maska tyl
		face.clear();
		face.push_back((Point)model2D.col(12) + Point(xs,ys));
		face.push_back((Point)model2D.col(13) + Point(xs,ys));
		face.push_back((Point)model2D.col(15) + Point(xs,ys));
		face.push_back((Point)model2D.col(14) + Point(xs,ys));
		faces.push_back(face);

		//sciana pionowa przod
		face.clear();
		face.push_back((Point)model2D.col(1) + Point(xs,ys));
		face.push_back((Point)model2D.col(2) + Point(xs,ys));
		face.push_back((Point)model2D.col(5) + Point(xs,ys));
		face.push_back((Point)model2D.col(4) + Point(xs,ys));
		faces.push_back(face);

		//sciana pionowa tyl
		face.clear();
		face.push_back((Point)model2D.col(0) + Point(xs,ys));
		face.push_back((Point)model2D.col(14) + Point(xs,ys));
		face.push_back((Point)model2D.col(15) + Point(xs,ys));
		face.push_back((Point)model2D.col(3) + Point(xs,ys));
		faces.push_back(face);

		//sciana pionowa bok1
		face.clear();
		face.push_back((Point)model2D.col(2) + Point(xs,ys));
		face.push_back((Point)model2D.col(3) + Point(xs,ys));
		face.push_back((Point)model2D.col(15) + Point(xs,ys));
		face.push_back((Point)model2D.col(13) + Point(xs,ys));
		face.push_back((Point)model2D.col(7) + Point(xs,ys));
		face.push_back((Point)model2D.col(5) + Point(xs,ys));
		faces.push_back(face);

		//sciana pionowa bok2
		face.clear();
		face.push_back((Point)model2D.col(0) + Point(xs,ys));
		face.push_back((Point)model2D.col(1) + Point(xs,ys));
		face.push_back((Point)model2D.col(4) + Point(xs,ys));
		face.push_back((Point)model2D.col(6) + Point(xs,ys));
		face.push_back((Point)model2D.col(12) + Point(xs,ys));
		face.push_back((Point)model2D.col(14) + Point(xs,ys));
		faces.push_back(face);

		//sciana skosna przod
		face.clear();
		face.push_back((Point)model2D.col(6) + Point(xs,ys));
		face.push_back((Point)model2D.col(7) + Point(xs,ys));
		face.push_back((Point)model2D.col(9) + Point(xs,ys));
		face.push_back((Point)model2D.col(8) + Point(xs,ys));
		faces.push_back(face);

		//sciana skosna tyl
		face.clear();
		face.push_back((Point)model2D.col(10) + Point(xs,ys));
		face.push_back((Point)model2D.col(12) + Point(xs,ys));
		face.push_back((Point)model2D.col(13) + Point(xs,ys));
		face.push_back((Point)model2D.col(11) + Point(xs,ys));
		faces.push_back(face);

		//sciana szyba bok1
		face.clear();
		face.push_back((Point)model2D.col(6) + Point(xs,ys));
		face.push_back((Point)model2D.col(8) + Point(xs,ys));
		face.push_back((Point)model2D.col(10) + Point(xs,ys));
		face.push_back((Point)model2D.col(12) + Point(xs,ys));
		faces.push_back(face);

		//sciana szyba bok2
		face.clear();
		face.push_back((Point)model2D.col(7) + Point(xs,ys));
		face.push_back((Point)model2D.col(9) + Point(xs,ys));
		face.push_back((Point)model2D.col(11) + Point(xs,ys));
		face.push_back((Point)model2D.col(13) + Point(xs,ys));
		faces.push_back(face);

		//maj¹c zbudowane wszystkie œciany, mo¿emy je narysowaæ po kolei
		for (int i = 0; i < faces.size(); i++)
			fillConvexPoly(img, faces[i], Scalar(255)); //rysuje jeden wielok¹t wypuk³y na raz i jest szybsza od fillPoly




	}

	//rysuje krawêdzie samochodu
	void drawEdges(Mat& img)
	{
		int g = 1;
		//podstawa
		line(img, (Point)model2D.col(0), (Point)model2D.col(1), Scalar(200), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(1), (Point)model2D.col(2), Scalar(200), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(2), (Point)model2D.col(3), Scalar(200), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(3), (Point)model2D.col(0), Scalar(200), g, LineTypes::LINE_AA);

		//pionowe
		line(img, (Point)model2D.col(1), (Point)model2D.col(4), Scalar(100), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(2), (Point)model2D.col(5), Scalar(100), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(3), (Point)model2D.col(15), Scalar(100), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(0), (Point)model2D.col(14), Scalar(100), g, LineTypes::LINE_AA);

		//wyzej poziome
		line(img, (Point)model2D.col(5), (Point)model2D.col(7), Scalar(50, 50, 50), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(7), (Point)model2D.col(13), Scalar(50, 50, 50), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(13), (Point)model2D.col(15), Scalar(50, 50, 50), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(15), (Point)model2D.col(14), Scalar(50, 50, 50), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(14), (Point)model2D.col(12), Scalar(50, 50, 50), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(12), (Point)model2D.col(6), Scalar(50, 50, 50), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(6), (Point)model2D.col(4), Scalar(50, 50, 50), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(4), (Point)model2D.col(5), Scalar(50, 50, 50), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(6), (Point)model2D.col(7), Scalar(50, 50, 50), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(12), (Point)model2D.col(13), Scalar(50, 50, 50), g, LineTypes::LINE_AA);

		//poziome góra
		line(img, (Point)model2D.col(8), (Point)model2D.col(9), Scalar(100, 0, 100), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(9), (Point)model2D.col(11), Scalar(100, 0, 100), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(11), (Point)model2D.col(10), Scalar(100, 0, 100), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(10), (Point)model2D.col(8), Scalar(100, 0, 100), g, LineTypes::LINE_AA);

		//skosy
		line(img, (Point)model2D.col(6), (Point)model2D.col(8), Scalar(0, 100, 100), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(7), (Point)model2D.col(9), Scalar(0, 100, 100), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(13), (Point)model2D.col(11), Scalar(0, 100, 100), g, LineTypes::LINE_AA);
		line(img, (Point)model2D.col(12), (Point)model2D.col(10), Scalar(0, 100, 100), g, LineTypes::LINE_AA);





	}

};


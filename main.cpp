#include <opencv2/opencv.hpp>
#include<fstream>
#include<chrono>
#include<omp.h>
#include<random>
#include "pso.h"
#include "car.h"
#include "carprosty.h"
#include "carnowy.h"
#include <iostream>
#include<locale>
#include <filesystem>

using namespace cv;
using namespace std;


//additional data to be passed to the optimization function
struct PSO_data
{
	Mat  input_img;;
	Mat t, r, macierzKamery, wspolczynnikiZnieksztalcen;
	Rect ramka;
	//add something if necessary
};


static double cost_Function(double* vec, int dim, void* params)
{
	//read passed parameters
	PSO_data* ddPSO = (PSO_data*)params;
	Mat input_img = ddPSO->input_img;
	Mat t = ddPSO->t;
	Mat r = ddPSO->r;
	Mat macierzKamery = ddPSO->macierzKamery;
	Mat wspolczynnikiZnieksztalcen = ddPSO->wspolczynnikiZnieksztalcen;
	Rect ramka = ddPSO->ramka;

	Mat img3 = Mat(input_img.size(), CV_8U, Scalar(0));


	//car
	new_car car1(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8], vec[9], vec[10], vec[11], vec[12]);
	car1.project(r, t, macierzKamery, wspolczynnikiZnieksztalcen);
	car1.draw(img3, -ramka.x, -ramka.y);


	//this is to be minimized
	//input_img - img3 - ile pikseli pozostanie widocznych po pokryciu maski naszym modelem - to jest za mało jaki kryterium
	//bo wystarcy wielki model na cały ekran i cała maska zostanie przykryta
	//img3 - input_img - jak bardzo nasz model wystaje poza maskę, chcemy by wystawał jak najmniej
	//jeśli rozpatrujemy tylko fragment obrazu, to nasz model może wyskoczyć poza ten fragment i wówczas, mimo że wystaje poza maskę
	//nie będziemy tego widzieć i algorytm uzna, że jest świetnie - innymi słowy zerowy wynik jest podejrzany i trzeba go eliminować
	//Mxor - suma dwóch powyższych
	//wyróżniając składowe Mxora możemy nadać im różne wagi, np. zezwolić by model trochę wystawał poza maskę, ale
	//aby żadne piksele maski nie wystawały poza model
	//czyli wystającym pikselom maski nadajemy większą wagę przez co bardziej wpływają na koszt
	//jednak same wystające piksele lub ich dominacja popsuje PSO bo większość cząsteczek przy kompletnie nietrafionym modelu
	//zwróci ten sam koszt i PSO zgłupieje
	int model_za_duzy = countNonZero(img3 - input_img);//piksele modelu wystające poza maskę
	int wystajaca_maska = countNonZero(input_img - img3); //piksele maski nieprzykryte przez model
	//double cost = 1*model_za_duzy+1*wystajaca_maska;//wystająca maska ma większą wagę, co powoduje że mod5l chce objąć całą maskę
	//double cost = 2*(vec[0]+vec[1]+vec[2]) +  5*wystajaca_maska;//suma wymiarów modelu ma być jak najmniejsza (jak najciaśniejsze pudło) i ma obejmować całą maskę
	//double cost = 10 * (vec[0] * vec[1] * vec[2]) + 10 * wystajaca_maska;//objętość modelu ma być jak najmniejsza (jak najciaśniejsze pudło) i ma obejmować całą maskę

	//double cost = 0 * (vec[0] * vec[1] * vec[2]) + 1 * model_za_duzy + 7 * wystajaca_maska;//wystająca maska ma większą wagę, co powoduje że mod5l chce objąć całą maskę


	//wystająca maska ma większą wagę, co powoduje że model chce objąć całą maskę
	//dodadkowe składniki dotyczą vec[5] - długość pokrywy silnika, vec[7] - długość dachu oraz vec[9] -szerokość dachu
	//są to największe elementy i zależy mi na ich dopasowaniu najlepiej
	//Długość dachu oraz długość maski przedniej posiadają wagi ujemne co oznacza, że chcemy maksymalizować te wartości.\
	 Natomiast szerokość dachu posiada wagę dodatnią co wiąże się z minimalizowaniem tego parametru
	double cost = 1 * model_za_duzy + 10 * wystajaca_maska- 2000 * vec[7] + 1000 * vec[9] - 2000 * vec[5]; 

	return cost;
}

static double cost_Function_model_prosty(double* vec, int dim, void* params)
{
	//read passed parameters
	PSO_data* ddPSO = (PSO_data*)params;
	Mat input_img = ddPSO->input_img;
	Mat t = ddPSO->t;
	Mat r = ddPSO->r;
	Mat macierzKamery = ddPSO->macierzKamery;
	Mat wspolczynnikiZnieksztalcen = ddPSO->wspolczynnikiZnieksztalcen;
	Rect ramka = ddPSO->ramka;

	Mat img3 = Mat(input_img.size(), CV_8U, Scalar(0));

	//PROSTY MODEL
	new_carprosty car2(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5]);
	car2.project(r, t, macierzKamery, wspolczynnikiZnieksztalcen);
	car2.draw(img3, -ramka.x, -ramka.y);

	//this is to be minimized
	//input_img - img3 - ile pikseli pozostanie widocznych po pokryciu maski naszym modelem - to jest za mało jaki kryterium
	//bo wystarcy wielki model na cały ekran i cała maska zostanie przykryta
	//img3 - input_img - jak bardzo nasz model wystaje poza maskę, chcemy by wystawał jak najmniej
	//jeśli rozpatrujemy tylko fragment obrazu, to nasz model może wyskoczyć poza ten fragment i wówczas, mimo że wystaje poza maskę
	//nie będziemy tego widzieć i algorytm uzna, że jest świetnie - innymi słowy zerowy wynik jest podejrzany i trzeba go eliminować
	//Mxor - suma dwóch powyższych
	//wyróżniając składowe Mxora możemy nadać im różne wagi, np. zezwolić by model trochę wystawał poza maskę, ale
	//aby żadne piksele maski nie wystawały poza model
	//czyli wystającym pikselom maski nadajemy większą wagę przez co bardziej wpływają na koszt
	//jednak same wystające piksele lub ich dominacja popsuje PSO bo większość cząsteczek przy kompletnie nietrafionym modelu
	//zwróci ten sam koszt i PSO zgłupieje
	int model_za_duzy = countNonZero(img3 - input_img);//piksele modelu wystające poza maskę
	int wystajaca_maska = countNonZero(input_img - img3); //piksele maski nieprzykryte przez model

	double cost = 10 * (vec[0] * vec[1] * vec[2]) + 10 * wystajaca_maska;//objętość modelu ma być jak najmniejsza (jak najciaśniejsze pudło) i ma obejmować całą maskę

	return cost;
}

int main()
{

	//deklaracja parametrów kamery
	Mat macierzKamery = (Mat_<double>(3, 3) << 1060.110459745811, 0, 969.5495542511281,
		0, 1090.02184615697, 569.1152750480022,
		0, 0, 1);

	Mat wspolczynnikiZnieksztalcen = (Mat_<double>(1, 8) <<
		-0.4324251772568071, 2.364132679815188, 0.008266725609657541, 0.001118370358531905, 1.556893153048111, 0.2409853472352748, 1.329097297275845, 3.677904323513058);

	Mat t = (Mat_<double>(3, 1) << -0.3553512619829435, -1.552620588350424, 15.9149979698209);

	Mat r = (Mat_<double>(3, 3) << 0.8811400433871717, -0.4727019129818488, -0.01204680052764304,
		-0.331550701349404, -0.5994611606427942, -0.7285056275112171,
		0.3371444147162614, 0.6459096053969894, -0.6849338838760534);


	//ładowanie zdjęć
	Mat  maskfromyolact, org;

	maskfromyolact = imread("fabia_m35.png", 0);
	org = imread("fabia_w35.png");

	
	Mat img8 = maskfromyolact;

	vector<vector<Point>> kontury;
	vector<Vec4i> hierarchia;
	vector<Rect>detectionBoxes;

	findContours(maskfromyolact, kontury, hierarchia, RETR_LIST, CHAIN_APPROX_SIMPLE);

	////usunięcie za małych  (bo czasem wykrywa malutki kawałek cienia)
	kontury.erase(remove_if(kontury.begin(), kontury.end(), [](const vector<Point>& kontur) {
		return contourArea(kontur) < 2000; }),
		kontury.end());
	drawContours(maskfromyolact, kontury, -1, CV_RGB(255, 0, 0), 2);

	Rect ramka = boundingRect(kontury[0]);

	//zwiększenie ramki o 50 w każdą stronę
	ramka.x -= 50;
	ramka.y -= 50;
	ramka.height += 100;
	ramka.width += 100;

	//wycięcie fragmentu zdjęcia z samochodem
	maskfromyolact = maskfromyolact(ramka);
	//imshow("test ramki", maskfromyolact);


	Mat kopia = org.clone(); //kopia obrazu - przyda się do rysowania siatki w pętli
	waitKey(1);


		//PSO optimization in the main function
		//use 12 cores to speed up optimization
		omp_set_num_threads(12);
		//set up optimization task - lower and upper bounds of all variables - that is enough

													//lx , ly,  lz,   x,    y,   b,   wf,     r,    wb,   wr,   hf,   hb, angle
		pso_settings_t* settings = pso_settings_new({ 3.7, 1.5, 1.3,  -15,  -10, 0.2,  0.15,  0.30,  0.08, 0.70, 0.5, 0.6,  0 },
			                                        { 5,   2,   1.55,  20,   25, 0.3,  0.20,  0.5,  0.15, 0.90, 0.6,  0.7, 360 },
			                                         0, 500, 500, 10, 20);
		
		
		// initialize GBEST solution and allocate memory for the best position buffer
		pso_result_t solution;
		solution.gbest = (double*)malloc(settings->dim * sizeof(double));

		//prepare additional data to be passed to the optimization function
		PSO_data function_params{ maskfromyolact, t, r, macierzKamery, wspolczynnikiZnieksztalcen, ramka };

		//run the optimization!
		auto begin = std::chrono::high_resolution_clock::now();
		pso_solve(*cost_Function, (void*)&function_params, &solution, settings);
		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

		cout << "SOLUTION:\n";
		cout << solution.gbest[0] << endl;
		cout << solution.gbest[1] << endl;
		cout << solution.gbest[2] << endl;
		cout << solution.gbest[3] << endl;
		cout << solution.gbest[4] << endl;
		cout << solution.gbest[5] << endl;
		cout << solution.gbest[6] << endl;
		cout << solution.gbest[7] << endl;
		cout << solution.gbest[8] << endl;
		cout << solution.gbest[9] << endl;
		cout << solution.gbest[10] << endl;
		cout << solution.gbest[11] << endl;
		cout << solution.gbest[12] << endl;
		
		ofstream myfile;
		myfile.open("data.csv");
	

		printf("Czas calkowity: %.3f s.\n", elapsed.count() * 1e-9);
		printf("Czas 1 iteracji: %.3f us.\n", elapsed.count() * 1e-3 * 1e-5);

		//verify

		Mat img5(org.size(), CV_8U, Scalar(0));

		//ZLOŻONY MODEL 
		new_car box2(solution.gbest[0], solution.gbest[1], solution.gbest[2], solution.gbest[3], solution.gbest[4], solution.gbest[5], solution.gbest[6], solution.gbest[7], solution.gbest[8], solution.gbest[9], solution.gbest[10], solution.gbest[11], solution.gbest[12]);
		box2.project(r, t, macierzKamery, wspolczynnikiZnieksztalcen);
		box2.draw(img5);
		box2.drawEdges(org);
		
		imshow("test", org);
		

		
		vector<float> best;
		for (int i = 0; i < 3; i++)
		best.push_back(solution.gbest[i]);
		for (int i = 5; i < 12; i++)
		best.push_back(solution.gbest[i]);

		//waitKey(0);

		vector<float> b; //najlepsze rozwiązanie z 1. iteracji
		for (int i = 0; i < 5; i++) //lx, ly, lz, x, y
		b.push_back(solution.gbest[i]);
		//zostaje kąt ze złożonego modelu
		b.push_back(solution.gbest[12]);

		//powtórz rozpoznanie w kolejnych ujęciach ze zbliżonymi parametrami do tych z pierwszej próby
		do
		{
			org = kopia.clone();

			for (int i = 0; i < 3; i++) //margines na wymiary pudła +- 10 cm, ale nie można przekroczyć ogólnych ograniczeń
			{
				if (b[i] - .1 > settings->range_lo[i]) settings->range_lo[i] = b[i] - .4;
				if (b[i] + .1 < settings->range_hi[i]) settings->range_hi[i] = b[i] + .4;
			}
			for (int i = 3; i < 5; i++) //margines na pozycję +- 1 m, ale nie można przekroczyć ogólnych ograniczeń
			{
				if (b[i] - 1 > settings->range_lo[i]) settings->range_lo[i] = b[i] - 4;
				if (b[i] + 1 < settings->range_hi[i]) settings->range_hi[i] = b[i] + 4;
			}
			//margines na kąt +- 3 stopnie - UWAGA! Przepisuję kąt ze złożonego modelu do prostego!
			//tu nie sprawdzam czy mieścimy się w limitach, bo chyba nie ma sensu

			settings->range_lo[5] = b[5] - 6;
			settings->range_hi[5] = b[5] + 6;

			settings->size = 100; //zmniejszamy liczbę cząstek
			settings->low_delta_count_limit = 20;
			settings->dim = 6; //dla modelu prostego jest 6 parametrow

			{
				//run the optimization!
				auto begin = std::chrono::high_resolution_clock::now();
				pso_solve(*cost_Function_model_prosty, (void*)&function_params, &solution, settings);
				auto end = std::chrono::high_resolution_clock::now();
				auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);



				cout << "SOLUTION:\n";
				cout << solution.gbest[0] << endl;
				cout << solution.gbest[1] << endl;
				cout << solution.gbest[2] << endl;
				cout << solution.gbest[3] << endl;
				cout << solution.gbest[4] << endl;
				cout << solution.gbest[5] << endl;

				printf("Czas calkowity: %.3f s.\n", elapsed.count() * 1e-9);
				printf("Czas 1 iteracji: %.3f us.\n", elapsed.count() * 1e-3 * 1e-5);

				//verify

				Mat img6(org.size(), CV_8U, Scalar(0));

				//PROSTY MODEL
				new_carprosty box3(solution.gbest[0], solution.gbest[1], solution.gbest[2], solution.gbest[3], solution.gbest[4], solution.gbest[5]);
				box3.project(r, t, macierzKamery, wspolczynnikiZnieksztalcen);
				box3.draw(img6);

				box3.drawEdges(org);
				org = org(ramka);
				//imshow("test2", org);

			}

		} while (waitKey(0) != 27);
	return 0;
}

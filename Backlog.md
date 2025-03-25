	1.	Analiza textului știrii
	•	Extrage textul principal al articolului.
	•	Împarte conținutul în propoziții și analizează structura acestora.
	2.	Clasificare Fake News vs. Real News
	•	Algoritmi de machine learning care determină dacă articolul este fake sau real.
	•	Modele bazate pe NLP care detectează pattern-uri comune în știrile false (ex: exagerări, lipsa surselor, limbaj emoțional).
	3.	Verificarea surselor
	•	Compară site-ul articolului cu o bază de date de surse verificate (Media Bias/Fact Check, Open Sources).
	•	Avertizează utilizatorul dacă sursa este cunoscută pentru știri false.
	4.	Comparare cu știri de încredere
	•	Caută subiecte similare pe site-uri de știri verificate și compară faptele prezentate.
	•	Notifică utilizatorul dacă există diferențe semnificative între versiuni.
	5. 	Analiză detaliată și raport explicativ
 	•	Generează un raport cu argumente pro și contra pentru credibilitatea articolului.
	•	Explică ce factori au influențat scorul final.
	•	Evidențiază limbajul emoțional, lipsa surselor și alte trăsături problematice.
	6.	Interfață Web pentru utilizatori
	•	Un UI unde utilizatorii pot introduce un link sau un text de articol pentru analiză.
	•	Oferă un scor de credibilitate (ex: 0% – complet fake, 100% – foarte credibil).
 	7. 	Acces la sursele folosite pentru verificare
  	•	Listează toate sursele folosite pentru verificare.
	•	Oferă link-uri către articolele relevante.
	•	Explică metodologia de selecție a surselor.
 	8.	 Sugestii de surse alternative
  	•	Creează un mecanism care sugerează surse alternative de încredere.
	•	Prioritizează surse diverse pentru o perspectivă echilibrată.
 	9.	 Setări avansate pentru administratori
  	•	Adaugă setări pentru pragurile scorului de credibilitate.
	•	Permite ajustarea sensibilității algoritmului.
 	10. 	Rezumat automatizat al articolelor
  	•	Utilizează NLP pentru a extrage esențialul din articol.
	•	Afișează principalele puncte sub formă de listă.
 	11.	 Istoricul verificărilor
  	•	Creează o pagină unde utilizatorul vede ultimele articole analizate.
	•	Permite ștergerea manuală a istoricului.
 	12. 	Raportarea analizelor incorecte
  	•	Adaugă un buton „Raportează evaluarea” în UI.
	•	Creează un sistem de feedback pentru ajustarea modelului.






Arhitectura Aplicației

	1.	Utilizatorul introduce un link sau un text de articol.
	2.	Backend-ul preia articolul și îl analizează:
	•	Tokenizare și curățare text.
	•	Detectarea sursei și verificarea în baza de date.
	•	Compararea conținutului cu știri similare.
	•	Analiza sentimentului și clasificarea fake vs. real.
	3.	Returnează utilizatorului un scor de credibilitate + argumente.


User Stories

	1.	Ca utilizator obișnuit, vreau să pot introduce un link către un articol de știri, astfel încât să primesc o evaluare a autenticității acestuia.
	2.	Ca utilizator, vreau să primesc un scor de credibilitate pentru un articol, astfel încât să știu cât de sigură este știrea.
	3.	Ca utilizator, vreau să știu dacă site-ul care publică articolul este de încredere, astfel încât să evaluez mai bine autenticitatea acestuia.
	4.	Ca utilizator, vreau să primesc recomandări de articole similare din surse de încredere, astfel încât să pot compara informațiile.
	5.	Ca utilizator, vreau să pot primi un raport detaliat despre factorii care influențează scorul de credibilitate al unei știri, astfel încât să înțeleg de ce este considerată falsă sau reală.
	6.	Ca utilizator, vreau să am o interfață simplă unde să pot introduce un articol pentru analiză, astfel încât să folosesc ușor aplicația.
	7.	Ca jurnalist, vreau să pot vedea sursele utilizate pentru verificarea unei știri, astfel încât să pot înțelege mai bine procesul de validare.
	8.	Ca utilizator, vreau să primesc sugestii de surse alternative credibile pentru un articol analizat, astfel încât să pot verifica informațiile din mai multe perspective.
	9.	Ca administrator, vreau să pot seta praguri personalizabile pentru scorul de fake news, astfel încât să decid când un articol este considerat problematic.
	10.	 Ca utilizator, vreau să văd un rezumat al articolului și punctele sale esențiale, astfel încât să pot decide rapid dacă merită să-l citesc în întregime.
 	11.	 Ca utilizator, vreau să pot vedea un istoric al articolelor pe care le-am verificat, astfel încât să pot reveni asupra analizelor anterioare.
  	12. 	Ca utilizator, vreau să pot raporta un articol care cred că a fost evaluat incorect, astfel încât echipa să poată reanaliza cazul.

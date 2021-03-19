%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB Code for epidemic simulations with the SIDARTHE model in the work
%
% Modelling the COVID-19 epidemic and implementation of population-wide interventions in Italy
% by Giulia Giordano, Franco Blanchini, Raffaele Bruno, Patrizio Colaneri, Alessandro Di Filippo, Angela Di Matteo, Marta Colaneri
%
% Giulia Giordano, April 5, 2020
% Contact: giulia.giordano@unitn.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function cost = sidarthe_comp(params1, params2, params3, params4, params5, params6, params1c, params2c, params3c, params4c, params5c, params6c)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % DATA
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Italian population
    popolazione=60e6;

    % Data 20 February - 5 April (46 days):
    % Total Cases
    CasiTotali = [3 20 79 132 219 322 400 650 888 1128 1694 2036 2502 3089 3858 4636 5883 7375 9172 10149 12462 15113 17660 21157 24747 27980 31506 35713 41035 47021 53578 59138 63927 69176 74386 80539 86498 92472 97689 101739 105792 110574 115242 119827 124632 128948]/popolazione; % D+R+T+E+H_diagnosticati
    % Deaths
    Deceduti = [0 1 2 2 5 10 12 17 21 29 34 52 79 107 148 197 233 366 463 631 827 1016 1266 1441 1809 2158 2503 2978 3405 4032 4825 5476 6077 6820 7503 8165 9134 10023 10779 11591 12428 13155 13915 14681 15362 15887]/popolazione; % E
    % Recovered
    Guariti = [0 0 0 1 1 1 3 45 46 50 83 149 160 276 414 523 589 622 724 1004 1045 1258 1439 1966 2335 2749 2941 4025 4440 5129 6072 7024 7432 8326 9362 10361 10950 12384 13030 14620 15729 16847 18278 19758 20996 21815]/popolazione; % H_diagnosticati
    % Currently Positive
    Positivi = [3 19 77 129 213 311 385 588 821 1049 1577 1835 2263 2706 3296 3916 5061 6387 7985 8514 10590 12839 14955 17750 20603 23073 26062 28710 33190 37860 42681 46638 50418 54030 57521 62013 66414 70065 73880 75528 77635 80572 83049 85388 88274 91246]/popolazione; % D+R+T

    % Data 23 February - 5 April (from day 4 to day 46)
    % Currently positive: isolated at home
    Isolamento_domiciliare = [49 91 162 221 284 412 543 798 927 1000 1065 1155 1060 1843 2180 2936 2599 3724 5036 6201 7860 9268 10197 11108 12090 14935 19185 22116 23783 26522 28697 30920 33648 36653 39533 42588 43752 45420 48134 50456 52579 55270 58320]/popolazione; %D
    % Currently positive: hospitalised
    Ricoverati_sintomi = [54 99 114 128 248 345 401 639 742 1034 1346 1790 2394 2651 3557 4316 5038 5838 6650 7426 8372 9663 11025 12894 14363 15757 16020 17708 19846 20692 21937 23112 24753 26029 26676 27386 27795 28192 28403 28540 28741 29010 28949]/popolazione; % R
    % Currently positive: ICU
    Terapia_intensiva = [26 23 35 36 56 64 105 140 166 229 295 351 462 567 650 733 877 1028 1153 1328 1518 1672 1851 2060 2257 2498 2655 2857 3009 3204 3396 3489 3612 3732 3856 3906 3981 4023 4035 4053 4068 3994 3977]/popolazione; %T


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PARAMETERS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Simulation horizon: CAN BE MODIFIED AT ONE'S WILL PROVIDED THAT IT IS AT
    % LEAST EQUAL TO THE NUMBER OF DAYS FOR WHICH DATA ARE AVAILABLE
    Orizzonte = 46;

    % Plot yes/no: SET TO 1 IF PDF FIGURES MUST BE GENERATED, 0 OTHERWISE
    generatefig = 1;
    plotPDF = 1;

    % Time-step for Euler discretisation of the continuous-time system
    step=0.01;

    % Transmission rate due to contacts with UNDETECTED asymptomatic infected
    alfa=[params1(1),params1c(1)];
    % Transmission rate due to contacts with DETECTED asymptomatic infected
    beta=[params1(2),params1c(2)];
    % Transmission rate due to contacts with UNDETECTED symptomatic infected
    gamma=[params1(3),params1c(3)];
    % Transmission rate due to contacts with DETECTED symptomatic infected
    delta=[params1(4),params1c(4)];

    % Detection rate for ASYMPTOMATIC
    epsilon=[params1(5),params1c(5)];
    % Detection rate for SYMPTOMATIC
    theta=[params1(6),params1c(6)];

    % Worsening rate: UNDETECTED asymptomatic infected becomes symptomatic
    zeta=[params1(7),params1c(7)];
    % Worsening rate: DETECTED asymptomatic infected becomes symptomatic
    eta=[params1(8),params1c(8)];

    % Worsening rate: UNDETECTED symptomatic infected develop life-threatening
    % symptoms
    mu=[params1(9),params1c(9)];
    % Worsening rate: DETECTED symptomatic infected develop life-threatening
    % symptoms
    nu=[params1(10),params1c(10)];

    % Mortality rate for infected with life-threatening symptoms
    tau=[params1(11),params1c(11)];

    % Recovery rate for undetected asymptomatic infected
    lambda=[params1(12),params1c(12)];
    % Recovery rate for detected asymptomatic infected
    rho=[params1(13),params1c(13)];
    % Recovery rate for undetected symptomatic infected
    kappa=[params1(14),params1c(14)];
    % Recovery rate for detected symptomatic infected
    xi=[params1(15),params1c(15)];
    % Recovery rate for life-threatened symptomatic infected
    sigma=[params1(16),params1c(16)];


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % DEFINITIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Parameters
    r1=epsilon+zeta+lambda;
    r2=eta+rho;
    r3=theta+mu+kappa;
    r4=nu+xi;
    r5=sigma+tau;

    % Initial R0
    R0_iniziale=alfa./r1+beta.*epsilon./(r1.*r2)+gamma.*zeta./(r1.*r3)+delta.*eta.*epsilon./(r1.*r2.*r4)+delta.*zeta.*theta./(r1.*r3.*r4);

    % Time horizon
    t=1:step:Orizzonte;

    % Vectors for time evolution of variables
    S=zeros(length(t), 2);
    I=zeros(length(t), 2);
    D=zeros(length(t), 2);
    A=zeros(length(t), 2);
    R=zeros(length(t), 2);
    T=zeros(length(t), 2);
    H=zeros(length(t), 2);
    H_diagnosticati=zeros(length(t), 2); % DIAGNOSED recovered only!
    E=zeros(length(t), 2);

    % Vectors for time evolution of actual/perceived Case Fatality Rate
    M=zeros(length(t), 2);
    P=zeros(length(t), 2);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % INITIAL CONDITIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    I(1, :)=200/popolazione;
    D(1, :)=20/popolazione;
    A(1, :)=1/popolazione;
    R(1, :)=2/popolazione;
    T(1, :)=0.00;
    H(1, :)=0.00;
    E(1, :)=0.00;
    S(1, :)=1-I(1,:)-D(1,:)-A(1,:)-R(1,:)-T(1,:)-H(1,:)-E(1,:);

    H_diagnosticati(1,:) = 0.00; % DIAGNOSED recovered only
    Infetti_reali(1,:)=I(1,:)+D(1,:)+A(1,:)+R(1,:)+T(1,:); % Actual currently infected

    M(1,:)=0;
    P(1,:)=0;

    % Whole state vector
    x=[S(1,:);I(1,:);D(1,:);A(1,:);R(1,:);T(1,:);H(1,:);E(1,:);H_diagnosticati(1,:);Infetti_reali(1,:)];

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % SIMULATION
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % "Control" binary variables to compute the new R0 every time a policy has
    % changed the parameters
    plottato = 0;
    plottato1 = 0;
    plottato_bis = 0;
    plottato_tris = 0;
    plottato_quat = 0;

    for i=2:length(t)

        if (i>4/step) % Basic social distancing (awareness, schools closed)
            alfa=[params2(1),params2c(1)];
            beta=[params2(2),params2c(2)];
            gamma=[params2(3),params2c(3)];
            delta=[params2(4),params2c(4)];
            if plottato == 0 % Compute the new R0
                r1=epsilon+zeta+lambda;
                r2=eta+rho;
                r3=theta+mu+kappa;
                r4=nu+xi;
                r5=sigma+tau;
                R0_primemisure=alfa./r1+beta.*epsilon./(r1.*r2)+gamma.*zeta./(r1.*r3)+delta.*eta.*epsilon./(r1.*r2.*r4)+delta.*zeta.*theta./(r1.*r3.*r4);
                plottato = 1;
            end
        end

        if (i>12/step)
            % Screening limited to / focused on symptomatic subjects
            epsilon=[params3(1),params3c(1)];
            if plottato1 == 0
                r1=epsilon+zeta+lambda;
                r2=eta+rho;
                r3=theta+mu+kappa;
                r4=nu+xi;
                r5=sigma+tau;
                R0_primemisureeps=alfa./r1+beta.*epsilon./(r1.*r2)+gamma.*zeta./(r1.*r3)+delta.*eta.*epsilon./(r1.*r2.*r4)+delta.*zeta.*theta./(r1.*r3.*r4);
                plottato1 = 1;
            end
        end

        if (i>22/step) % Social distancing: lockdown, mild effect

            alfa=[params4(1),params4c(1)];
            beta=[params4(2),params4c(2)];
            gamma=[params4(3),params4c(3)];
            delta=[params4(4),params4c(4)];

            mu=[params4(5),params4c(5)];
            nu=[params4(6),params4c(6)];

            zeta=[params4(7),params4c(7)];
            eta=[params4(8),params4c(8)];

            lambda=[params4(9),params4c(9)];
            rho=[params4(10),params4c(10)];
            kappa=[params4(11),params4c(11)];
            xi=[params4(12),params4c(12)];
            sigma=[params4(13),params4c(13)];

            if plottato_bis == 0 % Compute the new R0
                r1=epsilon+zeta+lambda;
                r2=eta+rho;
                r3=theta+mu+kappa;
                r4=nu+xi;
                r5=sigma+tau;
                R0_secondemisure=(alfa.*r2.*r3.*r4+epsilon.*beta.*r3.*r4+gamma.*zeta.*r2.*r4+delta.*eta.*epsilon.*r3+delta.*zeta.*theta.*r2)./(r1.*r2.*r3.*r4);
                plottato_bis = 1;
            end
        end

        if (i>28/step) % Social distancing: lockdown, strong effect

            alfa=[params5(1),params5c(1)];
            gamma=[params5(2),params5c(2)];

            if plottato_tris == 0 % Compute the new R0
                r1=epsilon+zeta+lambda;
                r2=eta+rho;
                r3=theta+mu+kappa;
                r4=nu+xi;
                r5=sigma+tau;
                R0_terzemisure=(alfa.*r2.*r3.*r4+epsilon.*beta.*r3.*r4+gamma.*zeta.*r2.*r4+delta.*eta.*epsilon.*r3+delta.*zeta.*theta.*r2)./(r1.*r2.*r3.*r4);
                plottato_tris = 1;
            end
        end

        if (i>38/step) % Broader diagnosis campaign

            epsilon=[params6(1),params6c(1)];
            rho=[params6(2),params6c(2)];
            kappa=[params6(3),params6c(3)];
            xi=[params6(4),params6c(4)];
            sigma=[params6(5),params6c(5)];

            zeta=[params6(6),params6c(6)];
            eta=[params6(7),params6c(7)];

            if plottato_quat == 0 % Compute the new R0
                r1=epsilon+zeta+lambda;
                r2=eta+rho;
                r3=theta+mu+kappa;
                r4=nu+xi;
                r5=sigma+tau;
                R0_quartemisure=(alfa.*r2.*r3.*r4+epsilon.*beta.*r3.*r4+gamma.*zeta.*r2.*r4+delta.*eta.*epsilon.*r3+delta.*zeta.*theta.*r2)./(r1.*r2.*r3.*r4);
                plottato_quat = 1;
            end
        end

        % Compute the system evolution

        B1=[-alfa(1)*x(2, 1)-beta(1)*x(3, 1)-gamma(1)*x(4, 1)-delta(1)*x(5, 1) 0 0 0 0 0 0 0 0 0;
            alfa(1)*x(2, 1)+beta(1)*x(3, 1)+gamma(1)*x(4, 1)+delta(1)*x(5, 1) -(epsilon+zeta+lambda)(1) 0 0 0 0 0 0 0 0;
            0 epsilon(1)  -(eta+rho)(1) 0 0 0 0 0 0 0;
            0 zeta(1) 0 -(theta+mu+kappa)(1) 0 0 0 0 0 0;
            0 0 eta(1) theta(1) -(nu+xi)(1) 0 0 0 0 0;
            0 0 0 mu(1) nu(1)  -(sigma+tau)(1) 0 0 0 0;
            0 lambda(1) rho(1) kappa(1) xi(1) sigma(1) 0 0 0 0;
            0 0 0 0 0 tau(1) 0 0 0 0;
            0 0 rho(1) 0 xi(1) sigma(1) 0 0 0 0;
            alfa(1)*x(2, 1)+beta(1)*x(3, 1)+gamma(1)*x(4, 1)+delta(1)*x(5, 1) 0 0 0 0 0 0 0 0 0];

        B2=[-alfa(2)*x(2, 2)-beta(2)*x(3, 2)-gamma(2)*x(4, 2)-delta(2)*x(5, 2) 0 0 0 0 0 0 0 0 0;
            alfa(2)*x(2, 2)+beta(2)*x(3, 2)+gamma(2)*x(4, 2)+delta(2)*x(5, 2) -(epsilon+zeta+lambda)(2) 0 0 0 0 0 0 0 0;
            0 epsilon(2)  -(eta+rho)(2) 0 0 0 0 0 0 0;
            0 zeta(2) 0 -(theta+mu+kappa)(2) 0 0 0 0 0 0;
            0 0 eta(2) theta(2) -(nu+xi)(2) 0 0 0 0 0;
            0 0 0 mu(2) nu(2)  -(sigma+tau)(2) 0 0 0 0;
            0 lambda(2) rho(2) kappa(2) xi(2) sigma(2) 0 0 0 0;
            0 0 0 0 0 tau(2) 0 0 0 0;
            0 0 rho(2) 0 xi(2) sigma(2) 0 0 0 0;
            alfa(2)*x(2, 2)+beta(2)*x(3, 2)+gamma(2)*x(4, 2)+delta(2)*x(5, 2) 0 0 0 0 0 0 0 0 0];

        x(:, 1)=x(:, 1)+B1*x(:, 1)*step;
        x(:, 2)=x(:, 2)+B2*x(:, 2)*step;

        % Update variables

        S(i, :)=x(1, :);
        I(i, :)=x(2, :);
        D(i, :)=x(3, :);
        A(i, :)=x(4, :);
        R(i, :)=x(5, :);
        T(i, :)=x(6, :);
        H(i, :)=x(7, :);
        E(i, :)=x(8, :);

        H_diagnosticati(i, :)=x(9, :);
        Infetti_reali(i, :)=x(10, :);

        % Update Case Fatality Rate

        M(i, :)=E(i, :)./(S(1, :)-S(i, :));
        P(i, :)=E(i, :)./((epsilon.*r3+(theta+mu).*zeta).*(I(1, :)+S(1, :)-I(i, :)-S(i, :))./(r1.*r3)+(theta+mu).*(A(1, :)-A(i, :))./r3);

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FINAL VALUES
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Variables
    Sbar=S(length(t));
    Ibar=I(length(t));
    Dbar=D(length(t));
    Abar=A(length(t));
    Rbar=R(length(t));
    Tbar=T(length(t));
    Hbar=H(length(t));
    Ebar=E(length(t));

    % Case fatality rate
    Mbar=M(length(t));
    Pbar=P(length(t));

    % Case fatality rate from formulas
    Mbar1=Ebar/(S(1)-Sbar);
    Pbar1=Ebar/((epsilon.*r3+(theta+mu).*zeta).*(I(1)+S(1)-Sbar-Ibar)/(r1.*r3)+(theta+mu).*(A(1)-Abar)/r3);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % COST
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    cost_h = zeros(1,2);
    j = 1;
    for i=1:1/step:size(CasiTotali,2)/step
        cost_h = cost_h + ( (H_diagnosticati(i, :)-Guariti(j)) * popolazione ).**2;
        j = j + 1;
    end

    j = 1;
    cost_d = zeros(1,2);
    cost_r = zeros(1,2);
    cost_t = zeros(1,2);
    for i=1+3/step:1/step:1+(size(Ricoverati_sintomi,2)+2)/step
        cost_d = cost_d + ( (D(i, :)-Isolamento_domiciliare(j)) .* popolazione ).**2;
        cost_r = cost_r + ( (R(i, :)-Ricoverati_sintomi(j)) .* popolazione ).**2;
        cost_t = cost_t + ( (T(i, :)-Terapia_intensiva(j)) .* popolazione ).**2;
        j = j + 1;
    end

    cost_d = cost_d/1e6
    cost_r = cost_r/1e6
    cost_t = cost_t/1e6
    cost_h = cost_h/1e6

    cost = cost_h + cost_d + cost_r + cost_t;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FIGURES
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if generatefig==1
        figure
        plot(t,Infetti_reali,'b',t,I+D+A+R+T,'r',t,H,'g',t,E,'k')
        hold on
        plot(t,D+R+T+E+H_diagnosticati,'--b',t,D+R+T,'--r',t,H_diagnosticati,'--g')
        xlim([t(1) t(end)])
        ylim([0 0.015])
        %title('Actual vs. Diagnosed Epidemic Evolution')
        xlabel('Time (days)')
        ylabel('Cases (fraction of the population)')
        legend({'Cumulative Infected','Current Total Infected', 'Recovered', 'Deaths','Diagnosed Cumulative Infected','Diagnosed Current Total Infected', 'Diagnosed Recovered'},'Location','northwest')
        grid

        if plotPDF==1
            set(gcf, 'PaperUnits', 'centimeters');
            set(gcf, 'PaperPosition', [0 0 24 16]);
            set(gcf, 'PaperSize', [24 16]); % dimension on x axis and y axis resp.
            print(gcf,'-dpdf', ['PanoramicaEpidemiaRealevsPercepita.pdf'])
        end
        %

        figure
        plot(t,I,'b',t,D,'c',t,A,'g',t,R,'m',t,T,'r')
        xlim([t(1) t(end)])
        ylim([0 1.1e-3])
        %title('Infected, different stages, Diagnosed vs. Non Diagnosed')
        xlabel('Time (days)')
        ylabel('Cases (fraction of the population)')
        legend({'Infected ND AS', 'Infected D AS', 'Infected ND S', 'Infected D S', 'Infected D IC'},'Location','northeast')
        grid

        if plotPDF==1
            set(gcf, 'PaperUnits', 'centimeters');
            set(gcf, 'PaperPosition', [0 0 24 16]);
            set(gcf, 'PaperSize', [24 16]); % dimension on x axis and y axis resp.
            print(gcf,'-dpdf', ['SuddivisioneInfetti.pdf'])
        end

        %

        figure
        plot(t,D+R+T+E+H_diagnosticati)
        hold on
        stem(t(1:1/step:size(CasiTotali,2)/step),CasiTotali)
        xlim([t(1) t(end)])
        ylim([0 2.5e-3])
        title('Cumulative Diagnosed Cases: Model vs. Data')
        xlabel('Time (days)')
        ylabel('Cases (fraction of the population)')
        grid

        if plotPDF==1
            set(gcf, 'PaperUnits', 'centimeters');
            set(gcf, 'PaperPosition', [0 0 16 10]);
            set(gcf, 'PaperSize', [16 10]); % dimension on x axis and y axis resp.
            print(gcf,'-dpdf', ['CasiTotali.pdf'])
        end
        %


        figure
        plot(t,H_diagnosticati)
        hold on
        stem(t(1:1/step:size(CasiTotali,2)/step),Guariti)
        xlim([t(1) t(end)])
        ylim([0 2.5e-3])
        title('Recovered: Model vs. Data')
        xlabel('Time (days)')
        ylabel('Cases (fraction of the population)')
        grid

        if plotPDF==1
            set(gcf, 'PaperUnits', 'centimeters');
            set(gcf, 'PaperPosition', [0 0 16 10]);
            set(gcf, 'PaperSize', [16 10]); % dimension on x axis and y axis resp.
            print(gcf,'-dpdf', ['Guariti_diagnosticati.pdf'])
        end
        %

        figure
        plot(t,E)
        hold on
        stem(t(1:1/step:size(CasiTotali,2)/step),Deceduti)
        xlim([t(1) t(end)])
        ylim([0 2.5e-3])
        title('Deaths: Model vs. Data - NOTE: EXCLUDED FROM FITTING')
        xlabel('Time (days)')
        ylabel('Cases (fraction of the population)')
        grid

        if plotPDF==1
            set(gcf, 'PaperUnits', 'centimeters');
            set(gcf, 'PaperPosition', [0 0 16 10]);
            set(gcf, 'PaperSize', [16 10]); % dimension on x axis and y axis resp.
            print(gcf,'-dpdf', ['Morti.pdf'])
        end
        %

        figure
        plot(t,D+R+T)
        hold on
        stem(t(1:1/step:size(CasiTotali,2)/step),Positivi)
        xlim([t(1) t(end)])
        ylim([0 2.5e-3])
        title('Infected: Model vs. Data')
        xlabel('Time (days)')
        ylabel('Cases (fraction of the population)')
        grid

        if plotPDF==1
            set(gcf, 'PaperUnits', 'centimeters');
            set(gcf, 'PaperPosition', [0 0 16 10]);
            set(gcf, 'PaperSize', [16 10]); % dimension on x axis and y axis resp.
            print(gcf,'-dpdf', ['Positivi_diagnosticati.pdf'])
        end
        %

        figure
        plot(t,D)
        hold on
        stem(t(1+3/step:1/step:1+(size(Ricoverati_sintomi,2)+2)/step),Isolamento_domiciliare)
        xlim([t(1) t(end)])
        ylim([0 2.5e-3])
        title('Infected, No Symptoms: Model vs. Data')
        xlabel('Time (days)')
        ylabel('Cases (fraction of the population)')
        grid

        if plotPDF==1
            set(gcf, 'PaperUnits', 'centimeters');
            set(gcf, 'PaperPosition', [0 0 16 10]);
            set(gcf, 'PaperSize', [16 10]); % dimension on x axis and y axis resp.
            print(gcf,'-dpdf', ['InfettiAsintomatici_diagnosticati.pdf'])
        end
        %

        figure
        plot(t,R)
        hold on
        stem(t(1+3/step:1/step:1+(size(Ricoverati_sintomi,2)+2)/step),Ricoverati_sintomi)
        xlim([t(1) t(end)])
        ylim([0 2.5e-3])
        title('Infected, Symptoms: Model vs. Data')
        xlabel('Time (days)')
        ylabel('Cases (fraction of the population)')
        grid

        if plotPDF==1
            set(gcf, 'PaperUnits', 'centimeters');
            set(gcf, 'PaperPosition', [0 0 16 10]);
            set(gcf, 'PaperSize', [16 10]); % dimension on x axis and y axis resp.
            print(gcf,'-dpdf', ['InfettiSintomatici_diagnosticati_ricoverati.pdf'])
        end
        %

        figure
        plot(t,D+R)
        hold on
        stem(t(1+3/step:1/step:1+(size(Ricoverati_sintomi,2)+2)/step),Isolamento_domiciliare+Ricoverati_sintomi)
        xlim([t(1) t(end)])
        ylim([0 2.5e-3])
        title('Infected, No or Mild Symptoms: Model vs. Data')
        xlabel('Time (days)')
        ylabel('Cases (fraction of the population)')
        grid

        if plotPDF==1
            set(gcf, 'PaperUnits', 'centimeters');
            set(gcf, 'PaperPosition', [0 0 16 10]);
            set(gcf, 'PaperSize', [16 10]); % dimension on x axis and y axis resp.
            print(gcf,'-dpdf', ['InfettiNonGravi_diagnosticati.pdf'])
        end

        %

        figure
        plot(t,T)
        hold on
        stem(t(1+3/step:1/step:1+(size(Ricoverati_sintomi,2)+2)/step),Terapia_intensiva)
        xlim([t(1) t(end)])
        ylim([0 2.5e-3])
        title('Infected, Life-Threatening Symptoms: Model vs. Data')
        xlabel('Time (days)')
        ylabel('Cases (fraction of the population)')
        grid

        if plotPDF==1
            set(gcf, 'PaperUnits', 'centimeters');
            set(gcf, 'PaperPosition', [0 0 16 10]);
            set(gcf, 'PaperSize', [16 10]); % dimension on x axis and y axis resp.
            print(gcf,'-dpdf', ['InfettiSintomatici_diagnosticati_terapiaintensiva.pdf'])
        end
    end
end

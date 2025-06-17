#import "@preview/zero:0.3.3": num

#figure(image("dtu_logo.svg", width: 20%), caption: [DTU Logo])
Aske Brunken s246123
== W.I.P
#set math.equation(numbering: "1")
#let data = csv("summary_statistics.csv")
= Describtive analysis
A) The data described in the output of "Simple summary of data" consists of 5
couloums, meaning 4 ETF, and a time Cloumn that tracks weeks. There are 453 rows
of recorded movement for the ETF's. The Data is collected from 2006-2015. Note
the data has more than the 4 ETF's described in this report but aren't analyzed.
Important to notice is that AGG is a bond based ETF while others are stock
based. Bonds are loans to typically to credit-worthy loaners, like govourments,
this typically results in lower volatility.


B) #table(
  columns: data.first().len(),
  ..data.first().map(c => [*#c*]),

  ..data
    .slice(1)
    .map(r => (
      [*#r.first()*],
      ..r.slice(1).map(n => num(n, digits: 6)),
    ))
    .flatten()
)

This table shows the summary statistics for the four ETFs (AGG, VAW, IWN, SPY):
- No_Obs: Number of observations
- Mean: Sample mean of weekly returns
- Variance: Sample variance
- Std_Dev: Standard deviation
- Q25, Q50, Q75: 25th, 50th (median), and 75th percentiles

C)



= Statistical analysis I
D) Covariance matrix:

#let cov_data = csv("covariance_table.csv")
#table(
  columns: cov_data.first().len(),
  ..cov_data.first().map(c => [*#c*]),
  ..cov_data
    .slice(1)
    .map(r => (
      [*#r.first()*],
      ..r.slice(1).map(n => num(n, digits: 6)),
    ))
    .flatten()
)
Here it is seen that AGG the bond ETF is very slightly negatively covaried with
the stock ETF's, this mainly shows that AGG isn't effected as dependant on the
stock market as stock ETF's. The stock based ETF's are positively covary, in
this case the intrepetration could be that the stock market thrives together,
when look deep into the data we will see that it also covaries negatively.


#figure(image("bbox_plots.png", width: 80%), caption: [Box plots of weekly ETF
  returns])
The greenline being in the positive is quite important for investment, it shows
that the typical return for a week is positive.

#figure(image("bempirical_density_plots.png", width: 80%), caption: [Empirical
  density plots of weekly ETF returns])

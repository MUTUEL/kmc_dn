Full boolean set search
=======================

This log will detail how I am looking for a full
boolean logic set in random layout 0.
In general I take the following parameters:

::
        kT = 1
        I_0 = 50*kT
        V_high = 500*kT
        ab_R = 0.25

Important to note, the controls range between [-V_high, V_high], whereas
the inputs range between [0, 2*V_high].

2018_12_06_141536_NAND
~~~~~~~~~~~~~~~~~~~~~~

Here I found a nice configuration. See the config for all parameters, but e.g.
a nice configuration was the following gene array([0.49336702, 0.5188791 , 0.67328908, 0.49878353, 0.54791179])
corresponding to electrode 3 ... 7.
See the plots below.

.. image:: 2018_12_06_141536_NAND/high_fitness.png
.. image:: 2018_12_06_141536_NAND/high_fitness_domain.png


XOR search
~~~~~~~~~~

Looking for XOR has proven to be quite difficult with the settings above.
Let's do another search on random layout 0, but allow much higher biases.
So both controls and inputs up to 2000kT.

I have evolved for I_0 and a_b as well. Will now try the following two combinations:

I_0 = 15kT
a_b = 0.22R || 0.8R


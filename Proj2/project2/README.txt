Adam Mertz
2295267
EECS 690
Proj 2

For my solution I read in each image and while reading broke the image down and sent (non blocking)
the size (int) followed by the image data (unsigned char) to the non-zero ranks. I used the blocking send
as I wanted to be sure they received the size data first before the raw image data. For the first image
I saved the Packed3DArray to a locally accessible variable. On the non-zero ranks I simply used 2
Recv calls to first grab the size and raw char data. From there I rebuild the Packed3DArray object
and used a function I wrote (CalculatedHistogram) to calculated the normalized 3x256 frequency array.
These were then flatted (Flatten2D function) and send back to the rank 0 process via a MPI_Gather call.
In the rank 0 process I then rebuilt these into a Nx768 array which was combined with the rank 0 histogram.
Having combined all the normalized histograms I then flattened the data back to a N * 768 array and used
MPI_Bcast to send the data to all processes. This probably could of been an immediate broadcast given that
I would not immediately need the data but given the difficulty I was having getting my application to work
I simply stuck with a blocking call. In each rank I think rebuilt the Nx3x256 array of data and calculated
the scores for each image in relation to all other images. This data was then flattened and returned back
to the rank 0 process where it was printed out to the user and used to calculate the image that was most
similar to the rank determined by the minimum score. As a sanity check our score when calculating the
difference between the image in rank 0 to rank N should be the same value as rank N to rank 0.
Unfortunately it does not appear that my application maintains this consistency. For example the score
of rank 0 to rank 1 is ~3.959 whereas the inverse is ~5.779. I have not been able to determine the cause.
Of the difference. I did find inconsistencies in the sum of the proportions at different stages in the
application. Where the actual differences are calcualted the values can vary between 0.95 and 1.01 whereas
when they are actually calculated (CalculateHistogram) they are consistently exactly 1. Whats more
interesting is that the array passed into "a" is always 1 and "b" is not. At some point the values must
of been slightly altered, possibly truncated, whether that was during transmission or another mathematical
operation I am unsure of.

- Adam

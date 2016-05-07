import sys
fname1 = sys.argv[1]
fname2 = sys.argv[2]
fp1 = open(fname1, "r")
fp2 = open(fname2, "r")

lines1 = fp1.readlines()
lines2 = fp2.readlines()

print "verifying..."

err_count = 0
err_accum = 0.0
for (x1, x2) in zip(lines1, lines2):
	xx1 = float(x1)
	xx2 = float(x2)

	diff = abs(xx1-xx2)
	maxof2 = max(abs(xx1), abs(xx2))
	if maxof2 != 0 and diff/maxof2 > 0.0001 :
		err_count += 1
		err_accum += diff


print "totally %d error(s) found, average error = %f" % (err_count, err_accum/len(lines1) )

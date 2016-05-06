fp1 = open("cpu.out", "r")
fp2 = open("gpu.out", "r")

lines1 = fp1.readlines()
lines2 = fp2.readlines()

print "verifying..."
i = 1
err_count = 0
err_accum = 0.0
for (x1, x2) in zip(lines1, lines2):
	xx1 = float(x1)
	xx2 = float(x2)

	diff = xx1 - xx2
	if diff < 0:
		diff = -diff
	if diff > 0.0001 or diff < -0.0001 :
		err_count += 1
		err_accum += diff

	i += 1

print "totally %d error(s) found, average error = %f" % (err_count, err_accum/len(lines1) )

class House():
    def __init__(self, line):
        # split the line into dimensions which will then be assigned to instance variable.
        # it's tedious and it does not look good, but I'm positive the added simplicity will be
        # beneficial later on.
        listDimensions = line.split(",")

        self.lotArea = listDimensions[0] # the area of the lot in feet square
        self.lotConfig = listDimensions[1] # configuration of the lotex: cul de sac, corner, inside, FR2(not sure what that one means)
        self.buildingType = listDimensions[2] # type of the building ex: 1 story, 2 story, 1.5Fin(finished?), 1.5Unf(Unfinished?), SFoyer
        self.houseStyle = listDimensions[3] # the style of the house ex: 2story, 1story etc.
        self.overallQuality = listDimensions[4] # a number from 0 to 10 giving the overall quality of the house in question
        self.overallCondition = listDimensions[5] # a number from 0 to 10 giving the overall condition of the house in question
        self.yearBuilt = listDimensions[6] # an integer depicting the year in which the house was built
        self.roofStyle = listDimensions[7] # the style of the roof ex: Gable, hip, gambrel, mansard, flat
        self.roofMaterial = listDimensions[8] # the material in which the roof was made ex: CompShg(CompactShingle), WdShg(WoodenShingles)
        self.exterior1st = listDimensions[9] # Primary material of the exterior ex: VinylSide, BrickFace, HdBoard(?), MetalSide
        self.exterior2nd = listDimensions[10] # Secondary material of the exterior ex: VinylSide, MetalSide, Plywood, BrickFace
        self.foundation = listDimensions[11] # The material of the foundation ex: CinderBlock, BrickTile, PConc(?Concrete), Wood, Slab
        self.basementQuality = listDimensions[12] # a qualifier giving the condition of the basement ex: Gd, Ex ...
        self.basementCondition = listDimensions[13] # a qualifier giving the condition of the basement ex: Gd, Ex ...
        self.basementExposure = listDimensions[14] # a qualifier for the amount of exposure in the basement ex: No, Mn(Minimal), Gd(Good), Av(Average)
        self.basementSurface = listDimensions[15] # the surface area of the basement in feet square
        self.heating = listDimensions[16] # the type of heating of the house ex: GasA(Gas?), GasW(GasWater?)
        self.centralAir = listDimensions[17] # if there is a central air system Y or N
        self.area1stFloor = listDimensions[18] # the area in square feet of the 1st floor
        self.area2stFloor = listDimensions[19]  # the area in square feet of the 2st floor
        self.basementFullBathroom = listDimensions[20] # the number of full bathrooms in the basement
        self.basementHalfBathroom = listDimensions[21] # the number of half bathrooms in the basement
        self.fullBathroom = listDimensions[22] # number of full bathrooms above the basement
        self.halfBathroom = listDimensions[23] # number of half bathrooms above the basement
        self.bedroomAboveGround = listDimensions[24] # number of bedrooms above ground
        self.kitchenAboveGround = listDimensions[25] # number of kitchen above the ground
        self.totalRoomsAboveGround = listDimensions[26] # total number of rooms above ground
        self.fireplaces = listDimensions[27] # number of fireplaces in the house
        self.garageType = listDimensions[28] # type of the garage ex: Attached, BuiltIn, Detached, CarPort, NA if no garage
        self.garageYearBuilt = listDimensions[29] # the year the garage was built, NA if no garage
        self.garageCars = listDimensions[30] # number of cars the garage can hold, 0 if no garage
        self.garageArea = listDimensions[31] # area in square feet of the garage, 0 is no garage
        self.pavedDriveway = listDimensions[32] # Y or N if the driveway is paved
        self.woodenDeckArea = listDimensions[33] # area in square feet of the wooden deck, 0 if no wooden deck
        self.openPorchArea = listDimensions[34] # the area of the open porch in square feet, 0 if no open porch
        self.enclosedPorchArea = listDimensions[35] # the area of the enclosed porch in square feet, 0 if no enclosed porch
        self.threeSeasonPorchArea = listDimensions[36] # the area of the 3 season porch in square feet, 0 if no 3 season porch
        self.screenPorchArea = listDimensions[37] # the area of the screen if there is one in feet square
        self.poolArea = listDimensions[38] # the area of the pool in square feet if there is one, 0 if no pool
        self.salePrice = listDimensions[39] # the price at which the house was sold, that is the label of our neural net.

        self.numCategorical = 15
        self.numNumerical = 25
        self.dimension = 40  # not including the sale price

    def categoricalData(self):
        categoricalData = []
        categoricalData.append(self.lotConfig)
        categoricalData.append(self.buildingType)
        categoricalData.append(self.houseStyle)
        categoricalData.append(self.roofStyle)
        categoricalData.append(self.roofMaterial)
        categoricalData.append(self.exterior1st)
        categoricalData.append(self.exterior2nd)
        categoricalData.append(self.foundation)
        categoricalData.append(self.basementQuality)
        categoricalData.append(self.basementCondition)
        categoricalData.append(self.basementExposure)
        categoricalData.append(self.heating)
        categoricalData.append(self.centralAir)
        categoricalData.append(self.garageType)
        categoricalData.append(self.pavedDriveway)

        return categoricalData

    def numberData(self):
        numberData = []
        numberData.append(int(self.lotArea))
        numberData.append(int(self.overallQuality))
        numberData.append(int(self.overallCondition))
        numberData.append(int(self.yearBuilt))
        numberData.append(int(self.basementSurface))
        numberData.append(int(self.area1stFloor))
        numberData.append(int(self.area2stFloor))
        numberData.append(int(self.basementFullBathroom))
        numberData.append(int(self.basementHalfBathroom))
        numberData.append(int(self.fullBathroom))
        numberData.append(int(self.halfBathroom))
        numberData.append(int(self.bedroomAboveGround))
        numberData.append(int(self.kitchenAboveGround))
        numberData.append(int(self.totalRoomsAboveGround))
        numberData.append(int(self.fireplaces))
        numberData.append(0 if self.garageYearBuilt == 'NA' else int(self.garageYearBuilt))
        numberData.append(int(self.garageCars))
        numberData.append(int(self.garageArea))
        numberData.append(int(self.woodenDeckArea))
        numberData.append(int(self.openPorchArea))
        numberData.append(int(self.enclosedPorchArea))
        numberData.append(int(self.threeSeasonPorchArea))
        numberData.append(int(self.screenPorchArea))
        numberData.append(int(self.poolArea))
        numberData.append((int(self.salePrice)))

        return numberData

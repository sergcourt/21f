       �K"	   ��Abrain.Event:2l��M�      ��	uX<��A"��
t
input/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@layer_1/weights1
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
VariableV2*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
layer_1/weights1/readIdentitylayer_1/weights1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
layer_1/biases1
VariableV2*"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
z
layer_1/biases1/readIdentitylayer_1/biases1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*'
_output_shapes
:���������*
T0
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@layer_2/weights2*
valueB"      
�
/layer_2/weights2/Initializer/random_uniform/minConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
layer_2/weights2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
layer_2/weights2/readIdentitylayer_2/weights2*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
!layer_2/biases2/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
layer_2/biases2
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*'
_output_shapes
:���������*
T0
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_3/weights3*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_3/weights3/Initializer/random_uniform/minConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *#
_class
loc:@layer_3/weights3*
valueB
 *׳]?
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3*
seed2 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
layer_3/weights3
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container 
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_3/weights3/readIdentitylayer_3/weights3*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
layer_3/biases3
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container 
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
z
layer_3/biases3/readIdentitylayer_3/biases3*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*'
_output_shapes
:���������*
T0
S
layer_3/ReluRelulayer_3/add*
T0*'
_output_shapes
:���������
�
0output/weights4/Initializer/random_uniform/shapeConst*"
_class
loc:@output/weights4*
valueB"      *
dtype0*
_output_shapes
:
�
.output/weights4/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *"
_class
loc:@output/weights4*
valueB
 *qĜ�
�
.output/weights4/Initializer/random_uniform/maxConst*"
_class
loc:@output/weights4*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
T0*"
_class
loc:@output/weights4*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
output/weights4
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
~
output/weights4/readIdentityoutput/weights4*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
w
output/biases4/readIdentityoutput/biases4*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
g

output/addAddoutput/MatMuloutput/biases4/read*'
_output_shapes
:���������*
T0
s
cost/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*
T0*'
_output_shapes
:���������
[

cost/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
n
$train/gradients/cost/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
&train/gradients/cost/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
_output_shapes
: *
T0
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*
T0*'
_output_shapes
:���������
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
_output_shapes
:*
T0*
out_type0
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/output/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul*'
_output_shapes
:���������
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*'
_output_shapes
:���������*
T0
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_3/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
_output_shapes
:*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
_output_shapes
:*
T0*
out_type0
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *"
_class
loc:@layer_1/biases1*
valueB
 *fff?
�
train/beta1_power
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: *
dtype0*
_output_shapes
: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta1_power/readIdentitytrain/beta1_power*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
�
train/beta2_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam
VariableV2*#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_1/weights1
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_1/weights1*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_1/weights1/Adam_1
VariableV2*#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
�
train/layer_1/biases1/Adam_1
VariableV2*"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container 
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam_1
VariableV2*#
_class
loc:@layer_2/weights2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam
VariableV2*"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_2/biases2
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_3/weights3*
valueB*    
�
train/layer_3/weights3/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
,train/output/weights4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*"
_class
loc:@output/weights4*
valueB*    
�
train/output/weights4/Adam
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container 
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
+train/output/biases4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam
VariableV2*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
�
-train/output/biases4/Adam_1/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container 
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
]
train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
T0*#
_class
loc:@layer_2/weights2*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_2/biases2*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_3/biases3*
use_nesterov( *
_output_shapes
:
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*!
_class
loc:@output/biases4
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@layer_1/biases1
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*%
valueB Blogging/current_cost*
dtype0*
_output_shapes
: 
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
T0*
_output_shapes
: 
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1
�
save/SaveV2/shape_and_slicesConst*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_1/biases1
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
use_locking(*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_2/weights2
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign"4�k�     ��d]	S�=��AJ܉
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.12.02v1.12.0-0-ga6d8ffae09��
t
input/PlaceholderPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
1layer_1/weights1/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_1/weights1*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_1/weights1/Initializer/random_uniform/minConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7��*
dtype0*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/maxConst*#
_class
loc:@layer_1/weights1*
valueB
 *�7�?*
dtype0*
_output_shapes
: 
�
9layer_1/weights1/Initializer/random_uniform/RandomUniformRandomUniform1layer_1/weights1/Initializer/random_uniform/shape*
T0*#
_class
loc:@layer_1/weights1*
seed2 *
dtype0*
_output_shapes

:*

seed 
�
/layer_1/weights1/Initializer/random_uniform/subSub/layer_1/weights1/Initializer/random_uniform/max/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes
: 
�
/layer_1/weights1/Initializer/random_uniform/mulMul9layer_1/weights1/Initializer/random_uniform/RandomUniform/layer_1/weights1/Initializer/random_uniform/sub*
_output_shapes

:*
T0*#
_class
loc:@layer_1/weights1
�
+layer_1/weights1/Initializer/random_uniformAdd/layer_1/weights1/Initializer/random_uniform/mul/layer_1/weights1/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
layer_1/weights1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
layer_1/weights1/AssignAssignlayer_1/weights1+layer_1/weights1/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
layer_1/weights1/readIdentitylayer_1/weights1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
!layer_1/biases1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
layer_1/biases1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
layer_1/biases1/AssignAssignlayer_1/biases1!layer_1/biases1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
z
layer_1/biases1/readIdentitylayer_1/biases1*
_output_shapes
:*
T0*"
_class
loc:@layer_1/biases1
�
layer_1/MatMulMatMulinput/Placeholderlayer_1/weights1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
j
layer_1/addAddlayer_1/MatMullayer_1/biases1/read*
T0*'
_output_shapes
:���������
S
layer_1/ReluRelulayer_1/add*
T0*'
_output_shapes
:���������
�
1layer_2/weights2/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_2/weights2*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_2/weights2/Initializer/random_uniform/minConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/maxConst*#
_class
loc:@layer_2/weights2*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_2/weights2/Initializer/random_uniform/RandomUniformRandomUniform1layer_2/weights2/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@layer_2/weights2*
seed2 *
dtype0*
_output_shapes

:
�
/layer_2/weights2/Initializer/random_uniform/subSub/layer_2/weights2/Initializer/random_uniform/max/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes
: 
�
/layer_2/weights2/Initializer/random_uniform/mulMul9layer_2/weights2/Initializer/random_uniform/RandomUniform/layer_2/weights2/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
+layer_2/weights2/Initializer/random_uniformAdd/layer_2/weights2/Initializer/random_uniform/mul/layer_2/weights2/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
layer_2/weights2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
layer_2/weights2/AssignAssignlayer_2/weights2+layer_2/weights2/Initializer/random_uniform*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
layer_2/weights2/readIdentitylayer_2/weights2*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
!layer_2/biases2/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
layer_2/biases2
VariableV2*"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
layer_2/biases2/AssignAssignlayer_2/biases2!layer_2/biases2/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
z
layer_2/biases2/readIdentitylayer_2/biases2*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
layer_2/MatMulMatMullayer_1/Relulayer_2/weights2/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_2/addAddlayer_2/MatMullayer_2/biases2/read*'
_output_shapes
:���������*
T0
S
layer_2/ReluRelulayer_2/add*
T0*'
_output_shapes
:���������
�
1layer_3/weights3/Initializer/random_uniform/shapeConst*#
_class
loc:@layer_3/weights3*
valueB"      *
dtype0*
_output_shapes
:
�
/layer_3/weights3/Initializer/random_uniform/minConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]�*
dtype0*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/maxConst*#
_class
loc:@layer_3/weights3*
valueB
 *׳]?*
dtype0*
_output_shapes
: 
�
9layer_3/weights3/Initializer/random_uniform/RandomUniformRandomUniform1layer_3/weights3/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*#
_class
loc:@layer_3/weights3*
seed2 
�
/layer_3/weights3/Initializer/random_uniform/subSub/layer_3/weights3/Initializer/random_uniform/max/layer_3/weights3/Initializer/random_uniform/min*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes
: 
�
/layer_3/weights3/Initializer/random_uniform/mulMul9layer_3/weights3/Initializer/random_uniform/RandomUniform/layer_3/weights3/Initializer/random_uniform/sub*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
+layer_3/weights3/Initializer/random_uniformAdd/layer_3/weights3/Initializer/random_uniform/mul/layer_3/weights3/Initializer/random_uniform/min*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
layer_3/weights3
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
layer_3/weights3/AssignAssignlayer_3/weights3+layer_3/weights3/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
layer_3/weights3/readIdentitylayer_3/weights3*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
!layer_3/biases3/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
layer_3/biases3
VariableV2*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:
�
layer_3/biases3/AssignAssignlayer_3/biases3!layer_3/biases3/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
z
layer_3/biases3/readIdentitylayer_3/biases3*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
layer_3/MatMulMatMullayer_2/Relulayer_3/weights3/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
j
layer_3/addAddlayer_3/MatMullayer_3/biases3/read*'
_output_shapes
:���������*
T0
S
layer_3/ReluRelulayer_3/add*'
_output_shapes
:���������*
T0
�
0output/weights4/Initializer/random_uniform/shapeConst*"
_class
loc:@output/weights4*
valueB"      *
dtype0*
_output_shapes
:
�
.output/weights4/Initializer/random_uniform/minConst*"
_class
loc:@output/weights4*
valueB
 *qĜ�*
dtype0*
_output_shapes
: 
�
.output/weights4/Initializer/random_uniform/maxConst*"
_class
loc:@output/weights4*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
�
8output/weights4/Initializer/random_uniform/RandomUniformRandomUniform0output/weights4/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:*

seed *
T0*"
_class
loc:@output/weights4*
seed2 
�
.output/weights4/Initializer/random_uniform/subSub.output/weights4/Initializer/random_uniform/max.output/weights4/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@output/weights4
�
.output/weights4/Initializer/random_uniform/mulMul8output/weights4/Initializer/random_uniform/RandomUniform.output/weights4/Initializer/random_uniform/sub*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
*output/weights4/Initializer/random_uniformAdd.output/weights4/Initializer/random_uniform/mul.output/weights4/Initializer/random_uniform/min*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
output/weights4
VariableV2*"
_class
loc:@output/weights4*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
output/weights4/AssignAssignoutput/weights4*output/weights4/Initializer/random_uniform*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
~
output/weights4/readIdentityoutput/weights4*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
 output/biases4/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
output/biases4
VariableV2*!
_class
loc:@output/biases4*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
output/biases4/AssignAssignoutput/biases4 output/biases4/Initializer/zeros*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:*
use_locking(
w
output/biases4/readIdentityoutput/biases4*
T0*!
_class
loc:@output/biases4*
_output_shapes
:
�
output/MatMulMatMullayer_3/Reluoutput/weights4/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
g

output/addAddoutput/MatMuloutput/biases4/read*'
_output_shapes
:���������*
T0
s
cost/PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
{
cost/SquaredDifferenceSquaredDifference
output/addcost/Placeholder*
T0*'
_output_shapes
:���������
[

cost/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
s
	cost/MeanMeancost/SquaredDifference
cost/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
X
train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
train/gradients/FillFilltrain/gradients/Shapetrain/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
}
,train/gradients/cost/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
&train/gradients/cost/Mean_grad/ReshapeReshapetrain/gradients/Fill,train/gradients/cost/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
z
$train/gradients/cost/Mean_grad/ShapeShapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
#train/gradients/cost/Mean_grad/TileTile&train/gradients/cost/Mean_grad/Reshape$train/gradients/cost/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
|
&train/gradients/cost/Mean_grad/Shape_1Shapecost/SquaredDifference*
T0*
out_type0*
_output_shapes
:
i
&train/gradients/cost/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
n
$train/gradients/cost/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
#train/gradients/cost/Mean_grad/ProdProd&train/gradients/cost/Mean_grad/Shape_1$train/gradients/cost/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
p
&train/gradients/cost/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
%train/gradients/cost/Mean_grad/Prod_1Prod&train/gradients/cost/Mean_grad/Shape_2&train/gradients/cost/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
j
(train/gradients/cost/Mean_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
�
&train/gradients/cost/Mean_grad/MaximumMaximum%train/gradients/cost/Mean_grad/Prod_1(train/gradients/cost/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
'train/gradients/cost/Mean_grad/floordivFloorDiv#train/gradients/cost/Mean_grad/Prod&train/gradients/cost/Mean_grad/Maximum*
_output_shapes
: *
T0
�
#train/gradients/cost/Mean_grad/CastCast'train/gradients/cost/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
&train/gradients/cost/Mean_grad/truedivRealDiv#train/gradients/cost/Mean_grad/Tile#train/gradients/cost/Mean_grad/Cast*'
_output_shapes
:���������*
T0
{
1train/gradients/cost/SquaredDifference_grad/ShapeShape
output/add*
T0*
out_type0*
_output_shapes
:
�
3train/gradients/cost/SquaredDifference_grad/Shape_1Shapecost/Placeholder*
T0*
out_type0*
_output_shapes
:
�
Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs1train/gradients/cost/SquaredDifference_grad/Shape3train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
2train/gradients/cost/SquaredDifference_grad/scalarConst'^train/gradients/cost/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
/train/gradients/cost/SquaredDifference_grad/mulMul2train/gradients/cost/SquaredDifference_grad/scalar&train/gradients/cost/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/subSub
output/addcost/Placeholder'^train/gradients/cost/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
1train/gradients/cost/SquaredDifference_grad/mul_1Mul/train/gradients/cost/SquaredDifference_grad/mul/train/gradients/cost/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/SumSum1train/gradients/cost/SquaredDifference_grad/mul_1Atrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
3train/gradients/cost/SquaredDifference_grad/ReshapeReshape/train/gradients/cost/SquaredDifference_grad/Sum1train/gradients/cost/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
1train/gradients/cost/SquaredDifference_grad/Sum_1Sum1train/gradients/cost/SquaredDifference_grad/mul_1Ctrain/gradients/cost/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
5train/gradients/cost/SquaredDifference_grad/Reshape_1Reshape1train/gradients/cost/SquaredDifference_grad/Sum_13train/gradients/cost/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
/train/gradients/cost/SquaredDifference_grad/NegNeg5train/gradients/cost/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
<train/gradients/cost/SquaredDifference_grad/tuple/group_depsNoOp0^train/gradients/cost/SquaredDifference_grad/Neg4^train/gradients/cost/SquaredDifference_grad/Reshape
�
Dtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependencyIdentity3train/gradients/cost/SquaredDifference_grad/Reshape=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*F
_class<
:8loc:@train/gradients/cost/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Ftrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency_1Identity/train/gradients/cost/SquaredDifference_grad/Neg=^train/gradients/cost/SquaredDifference_grad/tuple/group_deps*
T0*B
_class8
64loc:@train/gradients/cost/SquaredDifference_grad/Neg*'
_output_shapes
:���������
r
%train/gradients/output/add_grad/ShapeShapeoutput/MatMul*
T0*
out_type0*
_output_shapes
:
q
'train/gradients/output/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
5train/gradients/output/add_grad/BroadcastGradientArgsBroadcastGradientArgs%train/gradients/output/add_grad/Shape'train/gradients/output/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
#train/gradients/output/add_grad/SumSumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency5train/gradients/output/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
'train/gradients/output/add_grad/ReshapeReshape#train/gradients/output/add_grad/Sum%train/gradients/output/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
%train/gradients/output/add_grad/Sum_1SumDtrain/gradients/cost/SquaredDifference_grad/tuple/control_dependency7train/gradients/output/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
)train/gradients/output/add_grad/Reshape_1Reshape%train/gradients/output/add_grad/Sum_1'train/gradients/output/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
0train/gradients/output/add_grad/tuple/group_depsNoOp(^train/gradients/output/add_grad/Reshape*^train/gradients/output/add_grad/Reshape_1
�
8train/gradients/output/add_grad/tuple/control_dependencyIdentity'train/gradients/output/add_grad/Reshape1^train/gradients/output/add_grad/tuple/group_deps*
T0*:
_class0
.,loc:@train/gradients/output/add_grad/Reshape*'
_output_shapes
:���������
�
:train/gradients/output/add_grad/tuple/control_dependency_1Identity)train/gradients/output/add_grad/Reshape_11^train/gradients/output/add_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/add_grad/Reshape_1*
_output_shapes
:
�
)train/gradients/output/MatMul_grad/MatMulMatMul8train/gradients/output/add_grad/tuple/control_dependencyoutput/weights4/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
+train/gradients/output/MatMul_grad/MatMul_1MatMullayer_3/Relu8train/gradients/output/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
3train/gradients/output/MatMul_grad/tuple/group_depsNoOp*^train/gradients/output/MatMul_grad/MatMul,^train/gradients/output/MatMul_grad/MatMul_1
�
;train/gradients/output/MatMul_grad/tuple/control_dependencyIdentity)train/gradients/output/MatMul_grad/MatMul4^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@train/gradients/output/MatMul_grad/MatMul*'
_output_shapes
:���������
�
=train/gradients/output/MatMul_grad/tuple/control_dependency_1Identity+train/gradients/output/MatMul_grad/MatMul_14^train/gradients/output/MatMul_grad/tuple/group_deps*
T0*>
_class4
20loc:@train/gradients/output/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_3/Relu_grad/ReluGradReluGrad;train/gradients/output/MatMul_grad/tuple/control_dependencylayer_3/Relu*'
_output_shapes
:���������*
T0
t
&train/gradients/layer_3/add_grad/ShapeShapelayer_3/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_3/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_3/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_3/add_grad/Shape(train/gradients/layer_3/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_3/add_grad/SumSum*train/gradients/layer_3/Relu_grad/ReluGrad6train/gradients/layer_3/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
(train/gradients/layer_3/add_grad/ReshapeReshape$train/gradients/layer_3/add_grad/Sum&train/gradients/layer_3/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_3/add_grad/Sum_1Sum*train/gradients/layer_3/Relu_grad/ReluGrad8train/gradients/layer_3/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
*train/gradients/layer_3/add_grad/Reshape_1Reshape&train/gradients/layer_3/add_grad/Sum_1(train/gradients/layer_3/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_3/add_grad/tuple/group_depsNoOp)^train/gradients/layer_3/add_grad/Reshape+^train/gradients/layer_3/add_grad/Reshape_1
�
9train/gradients/layer_3/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_3/add_grad/Reshape2^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_3/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_3/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_3/add_grad/Reshape_12^train/gradients/layer_3/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_3/MatMul_grad/MatMulMatMul9train/gradients/layer_3/add_grad/tuple/control_dependencylayer_3/weights3/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
,train/gradients/layer_3/MatMul_grad/MatMul_1MatMullayer_2/Relu9train/gradients/layer_3/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_3/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_3/MatMul_grad/MatMul-^train/gradients/layer_3/MatMul_grad/MatMul_1
�
<train/gradients/layer_3/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_3/MatMul_grad/MatMul5^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_3/MatMul_grad/MatMul*'
_output_shapes
:���������
�
>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_3/MatMul_grad/MatMul_15^train/gradients/layer_3/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_3/MatMul_grad/MatMul_1*
_output_shapes

:
�
*train/gradients/layer_2/Relu_grad/ReluGradReluGrad<train/gradients/layer_3/MatMul_grad/tuple/control_dependencylayer_2/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_2/add_grad/ShapeShapelayer_2/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_2/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
�
6train/gradients/layer_2/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_2/add_grad/Shape(train/gradients/layer_2/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
$train/gradients/layer_2/add_grad/SumSum*train/gradients/layer_2/Relu_grad/ReluGrad6train/gradients/layer_2/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
(train/gradients/layer_2/add_grad/ReshapeReshape$train/gradients/layer_2/add_grad/Sum&train/gradients/layer_2/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_2/add_grad/Sum_1Sum*train/gradients/layer_2/Relu_grad/ReluGrad8train/gradients/layer_2/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_2/add_grad/Reshape_1Reshape&train/gradients/layer_2/add_grad/Sum_1(train/gradients/layer_2/add_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
�
1train/gradients/layer_2/add_grad/tuple/group_depsNoOp)^train/gradients/layer_2/add_grad/Reshape+^train/gradients/layer_2/add_grad/Reshape_1
�
9train/gradients/layer_2/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_2/add_grad/Reshape2^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*;
_class1
/-loc:@train/gradients/layer_2/add_grad/Reshape*'
_output_shapes
:���������
�
;train/gradients/layer_2/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_2/add_grad/Reshape_12^train/gradients/layer_2/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_2/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_2/MatMul_grad/MatMulMatMul9train/gradients/layer_2/add_grad/tuple/control_dependencylayer_2/weights2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
,train/gradients/layer_2/MatMul_grad/MatMul_1MatMullayer_1/Relu9train/gradients/layer_2/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:*
transpose_a(
�
4train/gradients/layer_2/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_2/MatMul_grad/MatMul-^train/gradients/layer_2/MatMul_grad/MatMul_1
�
<train/gradients/layer_2/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_2/MatMul_grad/MatMul5^train/gradients/layer_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@train/gradients/layer_2/MatMul_grad/MatMul
�
>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_2/MatMul_grad/MatMul_15^train/gradients/layer_2/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*?
_class5
31loc:@train/gradients/layer_2/MatMul_grad/MatMul_1
�
*train/gradients/layer_1/Relu_grad/ReluGradReluGrad<train/gradients/layer_2/MatMul_grad/tuple/control_dependencylayer_1/Relu*
T0*'
_output_shapes
:���������
t
&train/gradients/layer_1/add_grad/ShapeShapelayer_1/MatMul*
T0*
out_type0*
_output_shapes
:
r
(train/gradients/layer_1/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
�
6train/gradients/layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs&train/gradients/layer_1/add_grad/Shape(train/gradients/layer_1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
$train/gradients/layer_1/add_grad/SumSum*train/gradients/layer_1/Relu_grad/ReluGrad6train/gradients/layer_1/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
(train/gradients/layer_1/add_grad/ReshapeReshape$train/gradients/layer_1/add_grad/Sum&train/gradients/layer_1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
&train/gradients/layer_1/add_grad/Sum_1Sum*train/gradients/layer_1/Relu_grad/ReluGrad8train/gradients/layer_1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
*train/gradients/layer_1/add_grad/Reshape_1Reshape&train/gradients/layer_1/add_grad/Sum_1(train/gradients/layer_1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
�
1train/gradients/layer_1/add_grad/tuple/group_depsNoOp)^train/gradients/layer_1/add_grad/Reshape+^train/gradients/layer_1/add_grad/Reshape_1
�
9train/gradients/layer_1/add_grad/tuple/control_dependencyIdentity(train/gradients/layer_1/add_grad/Reshape2^train/gradients/layer_1/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@train/gradients/layer_1/add_grad/Reshape
�
;train/gradients/layer_1/add_grad/tuple/control_dependency_1Identity*train/gradients/layer_1/add_grad/Reshape_12^train/gradients/layer_1/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@train/gradients/layer_1/add_grad/Reshape_1*
_output_shapes
:
�
*train/gradients/layer_1/MatMul_grad/MatMulMatMul9train/gradients/layer_1/add_grad/tuple/control_dependencylayer_1/weights1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
,train/gradients/layer_1/MatMul_grad/MatMul_1MatMulinput/Placeholder9train/gradients/layer_1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
4train/gradients/layer_1/MatMul_grad/tuple/group_depsNoOp+^train/gradients/layer_1/MatMul_grad/MatMul-^train/gradients/layer_1/MatMul_grad/MatMul_1
�
<train/gradients/layer_1/MatMul_grad/tuple/control_dependencyIdentity*train/gradients/layer_1/MatMul_grad/MatMul5^train/gradients/layer_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@train/gradients/layer_1/MatMul_grad/MatMul
�
>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1Identity,train/gradients/layer_1/MatMul_grad/MatMul_15^train/gradients/layer_1/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@train/gradients/layer_1/MatMul_grad/MatMul_1*
_output_shapes

:
�
train/beta1_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
train/beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape: 
�
train/beta1_power/AssignAssigntrain/beta1_powertrain/beta1_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta1_power/readIdentitytrain/beta1_power*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
�
train/beta2_power/initial_valueConst*"
_class
loc:@layer_1/biases1*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
train/beta2_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *"
_class
loc:@layer_1/biases1*
	container 
�
train/beta2_power/AssignAssigntrain/beta2_powertrain/beta2_power/initial_value*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
z
train/beta2_power/readIdentitytrain/beta2_power*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
-train/layer_1/weights1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
�
train/layer_1/weights1/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:
�
"train/layer_1/weights1/Adam/AssignAssigntrain/layer_1/weights1/Adam-train/layer_1/weights1/Adam/Initializer/zeros*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
 train/layer_1/weights1/Adam/readIdentitytrain/layer_1/weights1/Adam*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
/train/layer_1/weights1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_1/weights1*
valueB*    
�
train/layer_1/weights1/Adam_1
VariableV2*
shared_name *#
_class
loc:@layer_1/weights1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
$train/layer_1/weights1/Adam_1/AssignAssigntrain/layer_1/weights1/Adam_1/train/layer_1/weights1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
"train/layer_1/weights1/Adam_1/readIdentitytrain/layer_1/weights1/Adam_1*
T0*#
_class
loc:@layer_1/weights1*
_output_shapes

:
�
,train/layer_1/biases1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_1/biases1*
valueB*    
�
train/layer_1/biases1/Adam
VariableV2*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_1/biases1/Adam/AssignAssigntrain/layer_1/biases1/Adam,train/layer_1/biases1/Adam/Initializer/zeros*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:*
use_locking(
�
train/layer_1/biases1/Adam/readIdentitytrain/layer_1/biases1/Adam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
.train/layer_1/biases1/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_1/biases1*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_1/biases1/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_1/biases1*
	container *
shape:
�
#train/layer_1/biases1/Adam_1/AssignAssigntrain/layer_1/biases1/Adam_1.train/layer_1/biases1/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
!train/layer_1/biases1/Adam_1/readIdentitytrain/layer_1/biases1/Adam_1*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
:
�
-train/layer_2/weights2/Adam/Initializer/zerosConst*
dtype0*
_output_shapes

:*#
_class
loc:@layer_2/weights2*
valueB*    
�
train/layer_2/weights2/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2*
	container *
shape
:
�
"train/layer_2/weights2/Adam/AssignAssigntrain/layer_2/weights2/Adam-train/layer_2/weights2/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
 train/layer_2/weights2/Adam/readIdentitytrain/layer_2/weights2/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_2/weights2
�
/train/layer_2/weights2/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_2/weights2*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_2/weights2/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_2/weights2
�
$train/layer_2/weights2/Adam_1/AssignAssigntrain/layer_2/weights2/Adam_1/train/layer_2/weights2/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
"train/layer_2/weights2/Adam_1/readIdentitytrain/layer_2/weights2/Adam_1*
T0*#
_class
loc:@layer_2/weights2*
_output_shapes

:
�
,train/layer_2/biases2/Adam/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam
VariableV2*"
_class
loc:@layer_2/biases2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name 
�
!train/layer_2/biases2/Adam/AssignAssigntrain/layer_2/biases2/Adam,train/layer_2/biases2/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
train/layer_2/biases2/Adam/readIdentitytrain/layer_2/biases2/Adam*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
.train/layer_2/biases2/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_2/biases2*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_2/biases2/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_2/biases2*
	container *
shape:
�
#train/layer_2/biases2/Adam_1/AssignAssigntrain/layer_2/biases2/Adam_1.train/layer_2/biases2/Adam_1/Initializer/zeros*
T0*"
_class
loc:@layer_2/biases2*
validate_shape(*
_output_shapes
:*
use_locking(
�
!train/layer_2/biases2/Adam_1/readIdentitytrain/layer_2/biases2/Adam_1*
T0*"
_class
loc:@layer_2/biases2*
_output_shapes
:
�
-train/layer_3/weights3/Adam/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3*
	container *
shape
:
�
"train/layer_3/weights3/Adam/AssignAssigntrain/layer_3/weights3/Adam-train/layer_3/weights3/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
 train/layer_3/weights3/Adam/readIdentitytrain/layer_3/weights3/Adam*
_output_shapes

:*
T0*#
_class
loc:@layer_3/weights3
�
/train/layer_3/weights3/Adam_1/Initializer/zerosConst*#
_class
loc:@layer_3/weights3*
valueB*    *
dtype0*
_output_shapes

:
�
train/layer_3/weights3/Adam_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *#
_class
loc:@layer_3/weights3
�
$train/layer_3/weights3/Adam_1/AssignAssigntrain/layer_3/weights3/Adam_1/train/layer_3/weights3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
"train/layer_3/weights3/Adam_1/readIdentitytrain/layer_3/weights3/Adam_1*
T0*#
_class
loc:@layer_3/weights3*
_output_shapes

:
�
,train/layer_3/biases3/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*"
_class
loc:@layer_3/biases3*
valueB*    
�
train/layer_3/biases3/Adam
VariableV2*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:*
dtype0*
_output_shapes
:
�
!train/layer_3/biases3/Adam/AssignAssigntrain/layer_3/biases3/Adam,train/layer_3/biases3/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
train/layer_3/biases3/Adam/readIdentitytrain/layer_3/biases3/Adam*
_output_shapes
:*
T0*"
_class
loc:@layer_3/biases3
�
.train/layer_3/biases3/Adam_1/Initializer/zerosConst*"
_class
loc:@layer_3/biases3*
valueB*    *
dtype0*
_output_shapes
:
�
train/layer_3/biases3/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@layer_3/biases3*
	container *
shape:
�
#train/layer_3/biases3/Adam_1/AssignAssigntrain/layer_3/biases3/Adam_1.train/layer_3/biases3/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_3/biases3
�
!train/layer_3/biases3/Adam_1/readIdentitytrain/layer_3/biases3/Adam_1*
T0*"
_class
loc:@layer_3/biases3*
_output_shapes
:
�
,train/output/weights4/Adam/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam
VariableV2*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container *
shape
:
�
!train/output/weights4/Adam/AssignAssigntrain/output/weights4/Adam,train/output/weights4/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
train/output/weights4/Adam/readIdentitytrain/output/weights4/Adam*
T0*"
_class
loc:@output/weights4*
_output_shapes

:
�
.train/output/weights4/Adam_1/Initializer/zerosConst*"
_class
loc:@output/weights4*
valueB*    *
dtype0*
_output_shapes

:
�
train/output/weights4/Adam_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *"
_class
loc:@output/weights4*
	container 
�
#train/output/weights4/Adam_1/AssignAssigntrain/output/weights4/Adam_1.train/output/weights4/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
!train/output/weights4/Adam_1/readIdentitytrain/output/weights4/Adam_1*
_output_shapes

:*
T0*"
_class
loc:@output/weights4
�
+train/output/biases4/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*!
_class
loc:@output/biases4*
valueB*    
�
train/output/biases4/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4
�
 train/output/biases4/Adam/AssignAssigntrain/output/biases4/Adam+train/output/biases4/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
train/output/biases4/Adam/readIdentitytrain/output/biases4/Adam*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
�
-train/output/biases4/Adam_1/Initializer/zerosConst*!
_class
loc:@output/biases4*
valueB*    *
dtype0*
_output_shapes
:
�
train/output/biases4/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@output/biases4*
	container *
shape:
�
"train/output/biases4/Adam_1/AssignAssigntrain/output/biases4/Adam_1-train/output/biases4/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
 train/output/biases4/Adam_1/readIdentitytrain/output/biases4/Adam_1*
_output_shapes
:*
T0*!
_class
loc:@output/biases4
]
train/Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
U
train/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
U
train/Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
W
train/Adam/epsilonConst*
valueB
 *w�+2*
dtype0*
_output_shapes
: 
�
,train/Adam/update_layer_1/weights1/ApplyAdam	ApplyAdamlayer_1/weights1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_1/weights1*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_1/biases1/ApplyAdam	ApplyAdamlayer_1/biases1train/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*"
_class
loc:@layer_1/biases1*
use_nesterov( *
_output_shapes
:
�
,train/Adam/update_layer_2/weights2/ApplyAdam	ApplyAdamlayer_2/weights2train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_2/weights2*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_2/biases2/ApplyAdam	ApplyAdamlayer_2/biases2train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_2/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@layer_2/biases2
�
,train/Adam/update_layer_3/weights3/ApplyAdam	ApplyAdamlayer_3/weights3train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon>train/gradients/layer_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*#
_class
loc:@layer_3/weights3*
use_nesterov( *
_output_shapes

:
�
+train/Adam/update_layer_3/biases3/ApplyAdam	ApplyAdamlayer_3/biases3train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon;train/gradients/layer_3/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@layer_3/biases3
�
+train/Adam/update_output/weights4/ApplyAdam	ApplyAdamoutput/weights4train/output/weights4/Adamtrain/output/weights4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon=train/gradients/output/MatMul_grad/tuple/control_dependency_1*
T0*"
_class
loc:@output/weights4*
use_nesterov( *
_output_shapes

:*
use_locking( 
�
*train/Adam/update_output/biases4/ApplyAdam	ApplyAdamoutput/biases4train/output/biases4/Adamtrain/output/biases4/Adam_1train/beta1_power/readtrain/beta2_power/readtrain/Adam/learning_ratetrain/Adam/beta1train/Adam/beta2train/Adam/epsilon:train/gradients/output/add_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0*!
_class
loc:@output/biases4
�
train/Adam/mulMultrain/beta1_power/readtrain/Adam/beta1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
_output_shapes
: *
T0*"
_class
loc:@layer_1/biases1
�
train/Adam/AssignAssigntrain/beta1_powertrain/Adam/mul*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking( 
�
train/Adam/mul_1Multrain/beta2_power/readtrain/Adam/beta2,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam*
T0*"
_class
loc:@layer_1/biases1*
_output_shapes
: 
�
train/Adam/Assign_1Assigntrain/beta2_powertrain/Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*"
_class
loc:@layer_1/biases1
�

train/AdamNoOp^train/Adam/Assign^train/Adam/Assign_1,^train/Adam/update_layer_1/biases1/ApplyAdam-^train/Adam/update_layer_1/weights1/ApplyAdam,^train/Adam/update_layer_2/biases2/ApplyAdam-^train/Adam/update_layer_2/weights2/ApplyAdam,^train/Adam/update_layer_3/biases3/ApplyAdam-^train/Adam/update_layer_3/weights3/ApplyAdam+^train/Adam/update_output/biases4/ApplyAdam,^train/Adam/update_output/weights4/ApplyAdam
n
logging/current_cost/tagsConst*
dtype0*
_output_shapes
: *%
valueB Blogging/current_cost
l
logging/current_costScalarSummarylogging/current_cost/tags	cost/Mean*
_output_shapes
: *
T0
a
logging/Merge/MergeSummaryMergeSummarylogging/current_cost*
N*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_sliceslayer_1/biases1layer_1/weights1layer_2/biases2layer_2/weights2layer_3/biases3layer_3/weights3output/biases4output/weights4train/beta1_powertrain/beta2_powertrain/layer_1/biases1/Adamtrain/layer_1/biases1/Adam_1train/layer_1/weights1/Adamtrain/layer_1/weights1/Adam_1train/layer_2/biases2/Adamtrain/layer_2/biases2/Adam_1train/layer_2/weights2/Adamtrain/layer_2/weights2/Adam_1train/layer_3/biases3/Adamtrain/layer_3/biases3/Adam_1train/layer_3/weights3/Adamtrain/layer_3/weights3/Adam_1train/output/biases4/Adamtrain/output/biases4/Adam_1train/output/weights4/Adamtrain/output/weights4/Adam_1*(
dtypes
2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�Blayer_1/biases1Blayer_1/weights1Blayer_2/biases2Blayer_2/weights2Blayer_3/biases3Blayer_3/weights3Boutput/biases4Boutput/weights4Btrain/beta1_powerBtrain/beta2_powerBtrain/layer_1/biases1/AdamBtrain/layer_1/biases1/Adam_1Btrain/layer_1/weights1/AdamBtrain/layer_1/weights1/Adam_1Btrain/layer_2/biases2/AdamBtrain/layer_2/biases2/Adam_1Btrain/layer_2/weights2/AdamBtrain/layer_2/weights2/Adam_1Btrain/layer_3/biases3/AdamBtrain/layer_3/biases3/Adam_1Btrain/layer_3/weights3/AdamBtrain/layer_3/weights3/Adam_1Btrain/output/biases4/AdamBtrain/output/biases4/Adam_1Btrain/output/weights4/AdamBtrain/output/weights4/Adam_1*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*(
dtypes
2*|
_output_shapesj
h::::::::::::::::::::::::::
�
save/AssignAssignlayer_1/biases1save/RestoreV2*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_1Assignlayer_1/weights1save/RestoreV2:1*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_2Assignlayer_2/biases2save/RestoreV2:2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_3Assignlayer_2/weights2save/RestoreV2:3*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_4Assignlayer_3/biases3save/RestoreV2:4*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assignlayer_3/weights3save/RestoreV2:5*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_6Assignoutput/biases4save/RestoreV2:6*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_7Assignoutput/weights4save/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/Assign_8Assigntrain/beta1_powersave/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: 
�
save/Assign_9Assigntrain/beta2_powersave/RestoreV2:9*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_10Assigntrain/layer_1/biases1/Adamsave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_11Assigntrain/layer_1/biases1/Adam_1save/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@layer_1/biases1*
validate_shape(*
_output_shapes
:
�
save/Assign_12Assigntrain/layer_1/weights1/Adamsave/RestoreV2:12*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_13Assigntrain/layer_1/weights1/Adam_1save/RestoreV2:13*
use_locking(*
T0*#
_class
loc:@layer_1/weights1*
validate_shape(*
_output_shapes

:
�
save/Assign_14Assigntrain/layer_2/biases2/Adamsave/RestoreV2:14*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_15Assigntrain/layer_2/biases2/Adam_1save/RestoreV2:15*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@layer_2/biases2
�
save/Assign_16Assigntrain/layer_2/weights2/Adamsave/RestoreV2:16*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_17Assigntrain/layer_2/weights2/Adam_1save/RestoreV2:17*
use_locking(*
T0*#
_class
loc:@layer_2/weights2*
validate_shape(*
_output_shapes

:
�
save/Assign_18Assigntrain/layer_3/biases3/Adamsave/RestoreV2:18*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_19Assigntrain/layer_3/biases3/Adam_1save/RestoreV2:19*
use_locking(*
T0*"
_class
loc:@layer_3/biases3*
validate_shape(*
_output_shapes
:
�
save/Assign_20Assigntrain/layer_3/weights3/Adamsave/RestoreV2:20*
use_locking(*
T0*#
_class
loc:@layer_3/weights3*
validate_shape(*
_output_shapes

:
�
save/Assign_21Assigntrain/layer_3/weights3/Adam_1save/RestoreV2:21*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*#
_class
loc:@layer_3/weights3
�
save/Assign_22Assigntrain/output/biases4/Adamsave/RestoreV2:22*
use_locking(*
T0*!
_class
loc:@output/biases4*
validate_shape(*
_output_shapes
:
�
save/Assign_23Assigntrain/output/biases4/Adam_1save/RestoreV2:23*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@output/biases4
�
save/Assign_24Assigntrain/output/weights4/Adamsave/RestoreV2:24*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@output/weights4
�
save/Assign_25Assigntrain/output/weights4/Adam_1save/RestoreV2:25*
use_locking(*
T0*"
_class
loc:@output/weights4*
validate_shape(*
_output_shapes

:
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^layer_1/biases1/Assign^layer_1/weights1/Assign^layer_2/biases2/Assign^layer_2/weights2/Assign^layer_3/biases3/Assign^layer_3/weights3/Assign^output/biases4/Assign^output/weights4/Assign^train/beta1_power/Assign^train/beta2_power/Assign"^train/layer_1/biases1/Adam/Assign$^train/layer_1/biases1/Adam_1/Assign#^train/layer_1/weights1/Adam/Assign%^train/layer_1/weights1/Adam_1/Assign"^train/layer_2/biases2/Adam/Assign$^train/layer_2/biases2/Adam_1/Assign#^train/layer_2/weights2/Adam/Assign%^train/layer_2/weights2/Adam_1/Assign"^train/layer_3/biases3/Adam/Assign$^train/layer_3/biases3/Adam_1/Assign#^train/layer_3/weights3/Adam/Assign%^train/layer_3/weights3/Adam_1/Assign!^train/output/biases4/Adam/Assign#^train/output/biases4/Adam_1/Assign"^train/output/weights4/Adam/Assign$^train/output/weights4/Adam_1/Assign""�
trainable_variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08"'
	summaries

logging/current_cost:0"
train_op


train/Adam"�
	variables��
w
layer_1/weights1:0layer_1/weights1/Assignlayer_1/weights1/read:02-layer_1/weights1/Initializer/random_uniform:08
j
layer_1/biases1:0layer_1/biases1/Assignlayer_1/biases1/read:02#layer_1/biases1/Initializer/zeros:08
w
layer_2/weights2:0layer_2/weights2/Assignlayer_2/weights2/read:02-layer_2/weights2/Initializer/random_uniform:08
j
layer_2/biases2:0layer_2/biases2/Assignlayer_2/biases2/read:02#layer_2/biases2/Initializer/zeros:08
w
layer_3/weights3:0layer_3/weights3/Assignlayer_3/weights3/read:02-layer_3/weights3/Initializer/random_uniform:08
j
layer_3/biases3:0layer_3/biases3/Assignlayer_3/biases3/read:02#layer_3/biases3/Initializer/zeros:08
s
output/weights4:0output/weights4/Assignoutput/weights4/read:02,output/weights4/Initializer/random_uniform:08
f
output/biases4:0output/biases4/Assignoutput/biases4/read:02"output/biases4/Initializer/zeros:08
l
train/beta1_power:0train/beta1_power/Assigntrain/beta1_power/read:02!train/beta1_power/initial_value:0
l
train/beta2_power:0train/beta2_power/Assigntrain/beta2_power/read:02!train/beta2_power/initial_value:0
�
train/layer_1/weights1/Adam:0"train/layer_1/weights1/Adam/Assign"train/layer_1/weights1/Adam/read:02/train/layer_1/weights1/Adam/Initializer/zeros:0
�
train/layer_1/weights1/Adam_1:0$train/layer_1/weights1/Adam_1/Assign$train/layer_1/weights1/Adam_1/read:021train/layer_1/weights1/Adam_1/Initializer/zeros:0
�
train/layer_1/biases1/Adam:0!train/layer_1/biases1/Adam/Assign!train/layer_1/biases1/Adam/read:02.train/layer_1/biases1/Adam/Initializer/zeros:0
�
train/layer_1/biases1/Adam_1:0#train/layer_1/biases1/Adam_1/Assign#train/layer_1/biases1/Adam_1/read:020train/layer_1/biases1/Adam_1/Initializer/zeros:0
�
train/layer_2/weights2/Adam:0"train/layer_2/weights2/Adam/Assign"train/layer_2/weights2/Adam/read:02/train/layer_2/weights2/Adam/Initializer/zeros:0
�
train/layer_2/weights2/Adam_1:0$train/layer_2/weights2/Adam_1/Assign$train/layer_2/weights2/Adam_1/read:021train/layer_2/weights2/Adam_1/Initializer/zeros:0
�
train/layer_2/biases2/Adam:0!train/layer_2/biases2/Adam/Assign!train/layer_2/biases2/Adam/read:02.train/layer_2/biases2/Adam/Initializer/zeros:0
�
train/layer_2/biases2/Adam_1:0#train/layer_2/biases2/Adam_1/Assign#train/layer_2/biases2/Adam_1/read:020train/layer_2/biases2/Adam_1/Initializer/zeros:0
�
train/layer_3/weights3/Adam:0"train/layer_3/weights3/Adam/Assign"train/layer_3/weights3/Adam/read:02/train/layer_3/weights3/Adam/Initializer/zeros:0
�
train/layer_3/weights3/Adam_1:0$train/layer_3/weights3/Adam_1/Assign$train/layer_3/weights3/Adam_1/read:021train/layer_3/weights3/Adam_1/Initializer/zeros:0
�
train/layer_3/biases3/Adam:0!train/layer_3/biases3/Adam/Assign!train/layer_3/biases3/Adam/read:02.train/layer_3/biases3/Adam/Initializer/zeros:0
�
train/layer_3/biases3/Adam_1:0#train/layer_3/biases3/Adam_1/Assign#train/layer_3/biases3/Adam_1/read:020train/layer_3/biases3/Adam_1/Initializer/zeros:0
�
train/output/weights4/Adam:0!train/output/weights4/Adam/Assign!train/output/weights4/Adam/read:02.train/output/weights4/Adam/Initializer/zeros:0
�
train/output/weights4/Adam_1:0#train/output/weights4/Adam_1/Assign#train/output/weights4/Adam_1/read:020train/output/weights4/Adam_1/Initializer/zeros:0
�
train/output/biases4/Adam:0 train/output/biases4/Adam/Assign train/output/biases4/Adam/read:02-train/output/biases4/Adam/Initializer/zeros:0
�
train/output/biases4/Adam_1:0"train/output/biases4/Adam_1/Assign"train/output/biases4/Adam_1/read:02/train/output/biases4/Adam_1/Initializer/zeros:0��A�(       �pJ	�a@��A*

logging/current_costP��=C��.*       ����	��@��A*

logging/current_cost4��=�,o*       ����	��@��A
*

logging/current_cost��=�
��*       ����	�@��A*

logging/current_cost�:�=�{�*       ����	�+A��A*

logging/current_cost-�=w$P�*       ����	�[A��A*

logging/current_cost�f="?�G*       ����	ߎA��A*

logging/current_cost��I=�Tf *       ����	��A��A#*

logging/current_cost�B3=vAV�*       ����	��A��A(*

logging/current_cost�!=v�#�*       ����	�*B��A-*

logging/current_cost��=�M2*       ����	[aB��A2*

logging/current_costLt=�`�*       ����	��B��A7*

logging/current_cost�i�<��*       ����	|�B��A<*

logging/current_cost$:�<)��0*       ����	��B��AA*

logging/current_cost��<�} �*       ����	$C��AF*

logging/current_cost���<4~��*       ����	?TC��AK*

logging/current_costz��<,4Ք*       ����	��C��AP*

logging/current_cost���<q}�*       ����	�C��AU*

logging/current_cost�ݮ<��[*       ����	��C��AZ*

logging/current_cost4�<r���*       ����	D��A_*

logging/current_costqQ�<��O*       ����	�:D��Ad*

logging/current_cost�i�<V;��*       ����	IjD��Ai*

logging/current_costnL�<j3F�*       ����	�D��An*

logging/current_cost�Ӈ<��&*       ����	��D��As*

logging/current_costDŁ<c�;**       ����	��D��Ax*

logging/current_cost59x<��Ś*       ����	1$E��A}*

logging/current_cost�5n<K�k+       ��K	oQE��A�*

logging/current_costD�d<��;+       ��K	��E��A�*

logging/current_cost�[<�rG+       ��K	q�E��A�*

logging/current_cost"�S<j��+       ��K	4�E��A�*

logging/current_cost��K<D�s�+       ��K	RF��A�*

logging/current_cost�D<��n�+       ��K	3;F��A�*

logging/current_cost�><���q+       ��K	hF��A�*

logging/current_cost' 8<v��+       ��K	��F��A�*

logging/current_cost�V2<V�.�+       ��K	7�F��A�*

logging/current_cost-<Ev�+       ��K	�F��A�*

logging/current_cost�/(<Eb�^+       ��K	{ G��A�*

logging/current_costD�#<{_O+       ��K	NG��A�*

logging/current_cost5�<Ʈգ+       ��K	�{G��A�*

logging/current_cost|<��`�+       ��K	�G��A�*

logging/current_cost@�<�pk�+       ��K	��G��A�*

logging/current_costj�<@��Q+       ��K	rH��A�*

logging/current_cost�<Qۂ�+       ��K	�<H��A�*

logging/current_cost�)<p
�=+       ��K	�jH��A�*

logging/current_cost��<�5S+       ��K	%�H��A�*

logging/current_cost�<4�+       ��K	��H��A�*

logging/current_cost�
<���+       ��K	cI��A�*

logging/current_cost~<���+       ��K	@/I��A�*

logging/current_cost*<��"5+       ��K	(]I��A�*

logging/current_cost��<<dY�+       ��K	܌I��A�*

logging/current_cost}�<��_�+       ��K	ɻI��A�*

logging/current_cost�<58�q+       ��K	��I��A�*

logging/current_cost.�<�_MJ+       ��K	&J��A�*

logging/current_costR�<Z?+       ��K	PJ��A�*

logging/current_costE3<p���+       ��K	�~J��A�*

logging/current_costD| <r2-+       ��K	��J��A�*

logging/current_cost���;FHp+       ��K	\�J��A�*

logging/current_cost���;x��+       ��K	K��A�*

logging/current_cost���;�(�+       ��K	�:K��A�*

logging/current_cost;��;��+       ��K	�gK��A�*

logging/current_cost��;^L+       ��K	9�K��A�*

logging/current_costr^�;���+       ��K	I�K��A�*

logging/current_cost���;�9��+       ��K	^�K��A�*

logging/current_costB�;K�T+       ��K	�#L��A�*

logging/current_cost���;���+       ��K	�QL��A�*

logging/current_cost�=�;.}�+       ��K	AL��A�*

logging/current_cost��;��=F+       ��K	+�L��A�*

logging/current_cost�p�;'�J�+       ��K	��L��A�*

logging/current_cost��;�q�q+       ��K	0M��A�*

logging/current_cost���;�O+       ��K	�7M��A�*

logging/current_cost9L�;@u+       ��K	�eM��A�*

logging/current_cost���;��|�+       ��K	L�M��A�*

logging/current_costt��;�ı�+       ��K	[�M��A�*

logging/current_cost��;����+       ��K	��M��A�*

logging/current_cost�h�;�Q�+       ��K	a(N��A�*

logging/current_costR2�;�p8
+       ��K	�VN��A�*

logging/current_costn��;���x+       ��K	p�N��A�*

logging/current_cost���;�E�w+       ��K	�N��A�*

logging/current_costR��;W��u+       ��K	Z�N��A�*

logging/current_costkk�;p�;+       ��K	O��A�*

logging/current_cost�D�;�.��+       ��K	�DO��A�*

logging/current_cost��;U�?+       ��K	�O��A�*

logging/current_cost���;',��+       ��K	.�O��A�*

logging/current_costu��;�SF�+       ��K	3P��A�*

logging/current_costG��;}+       ��K	QJP��A�*

logging/current_cost0��;�Ѯ+       ��K	T�P��A�*

logging/current_cost�[�;2��+       ��K	��P��A�*

logging/current_cost�.�;-��+       ��K	��P��A�*

logging/current_cost��;�8�+       ��K	�5Q��A�*

logging/current_cost^��;cF0�+       ��K	OmQ��A�*

logging/current_cost2��;RV��+       ��K	�Q��A�*

logging/current_costD}�;�m?�+       ��K	��Q��A�*

logging/current_cost�`�;���+       ��K	JR��A�*

logging/current_cost�O�;�7N�+       ��K	9JR��A�*

logging/current_cost)E�;�]cT+       ��K	4|R��A�*

logging/current_cost�2�;��#�+       ��K	αR��A�*

logging/current_costG�;7�+       ��K	��R��A�*

logging/current_cost��;iIp+       ��K	�S��A�*

logging/current_cost���;�若+       ��K	yBS��A�*

logging/current_cost��;�\R+       ��K	<xS��A�*

logging/current_cost���;���+       ��K	حS��A�*

logging/current_costĕ�;�P��+       ��K	��S��A�*

logging/current_cost5t�;�\w�+       ��K	�T��A�*

logging/current_cost�V�;'�p"+       ��K	�HT��A�*

logging/current_cost�<�;��+       ��K	�|T��A�*

logging/current_cost�$�;�AJ+       ��K	3�T��A�*

logging/current_costp�;m$aW+       ��K	{�T��A�*

logging/current_cost��;����+       ��K	5U��A�*

logging/current_cost���;iS$w+       ��K	}BU��A�*

logging/current_cost���;��5�+       ��K	�qU��A�*

logging/current_cost���;���+       ��K	�U��A�*

logging/current_cost���;k�s�+       ��K	�U��A�*

logging/current_cost���;�CbB+       ��K	>V��A�*

logging/current_cost���;����+       ��K	ZKV��A�*

logging/current_cost���;SI�+       ��K	'{V��A�*

logging/current_costyu�;,~+�+       ��K	R�V��A�*

logging/current_costm�;��-�+       ��K	��V��A�*

logging/current_costGl�;�:m�+       ��K	8W��A�*

logging/current_cost�m�;���+       ��K	w=W��A�*

logging/current_cost�f�;.1�+       ��K	�lW��A�*

logging/current_cost�\�;��n�+       ��K	�W��A�*

logging/current_cost�R�;�ϥ{+       ��K	��W��A�*

logging/current_cost�M�;��<+       ��K	YX��A�*

logging/current_cost�K�;�8E+       ��K	�/X��A�*

logging/current_cost;H�;0���+       ��K	+_X��A�*

logging/current_costC�;+��+       ��K	��X��A�*

logging/current_cost�&�;�7�q+       ��K	/�X��A�*

logging/current_cost�*�;W�*�+       ��K	5�X��A�*

logging/current_cost�>�;+��q+       ��K	�Y��A�*

logging/current_costL�;�DXq+       ��K	PMY��A�*

logging/current_cost�H�;�i�4+       ��K	�yY��A�*

logging/current_cost<�;w4�+       ��K	�Y��A�*

logging/current_costY,�;(�-�+       ��K	��Y��A�*

logging/current_cost��;���+       ��K	@
Z��A�*

logging/current_cost��;��hh+       ��K	"9Z��A�*

logging/current_cost�d�;�C=�+       ��K	OfZ��A�*

logging/current_cost�1�;�Z��+       ��K	d�Z��A�*

logging/current_cost���;�K�+       ��K	y�Z��A�*

logging/current_cost��;��Ct+       ��K	��Z��A�*

logging/current_cost�d�;��h+       ��K	�,[��A�*

logging/current_costd,�;��ɽ+       ��K	pZ[��A�*

logging/current_costK��;�*S�+       ��K	��[��A�*

logging/current_cost���; Ji�+       ��K	��[��A�*

logging/current_cost���;M35Z+       ��K	��[��A�*

logging/current_cost\��;�[j+       ��K	�\��A�*

logging/current_cost���;�eV+       ��K	OH\��A�*

logging/current_costˏ�;��~�+       ��K	�x\��A�*

logging/current_cost�w�;^��P+       ��K	��\��A�*

logging/current_cost�g�;�ց�+       ��K	��\��A�*

logging/current_costW�;��+       ��K	�]��A�*

logging/current_cost9E�;b��v+       ��K	�8]��A�*

logging/current_cost�/�;�uF�+       ��K	g]��A�*

logging/current_cost�;�O.�+       ��K	��]��A�*

logging/current_cost���;PƖ+       ��K	��]��A�*

logging/current_costK��;��+       ��K	7�]��A�*

logging/current_cost���;���+       ��K	�!^��A�*

logging/current_cost���;YPH+       ��K	�S^��A�*

logging/current_cost���;E8�+       ��K	��^��A�*

logging/current_cost���;�yh�+       ��K	ܴ^��A�*

logging/current_cost^��;�y�d+       ��K	��^��A�*

logging/current_cost���;k���+       ��K	�_��A�*

logging/current_cost���;�&E2+       ��K	�D_��A�*

logging/current_cost���;�%�)+       ��K	�t_��A�*

logging/current_cost���;��{+       ��K	6�_��A�*

logging/current_cost���;���+       ��K	��_��A�*

logging/current_cost���;�<H�+       ��K	��_��A�*

logging/current_cost.��;�7��+       ��K	�.`��A�*

logging/current_cost���;%$��+       ��K	�]`��A�*

logging/current_cost԰�;,�,�+       ��K	��`��A�*

logging/current_cost��;�+       ��K	6�`��A�*

logging/current_cost���;�}$+       ��K	C�`��A�*

logging/current_cost��;��+       ��K	 a��A�*

logging/current_cost���;$�+       ��K	�Ia��A�*

logging/current_costn��;6m�+       ��K	�wa��A�*

logging/current_costn��;zqC+       ��K	,�a��A�*

logging/current_costN��;F��+       ��K	�a��A�*

logging/current_cost���;��:+       ��K	O b��A�*

logging/current_cost���;)��+       ��K	A-b��A�*

logging/current_cost`��;Rr+       ��K	�[b��A�*

logging/current_cost.��;X�@j+       ��K	0�b��A�*

logging/current_cost���;��5+       ��K	|�b��A�*

logging/current_cost���;���+       ��K	h�b��A�*

logging/current_costP��;���+       ��K	�c��A�*

logging/current_cost���;�>��+       ��K	�Cc��A�*

logging/current_cost���; `g�+       ��K	�qc��A�*

logging/current_cost��;���+       ��K	��c��A�*

logging/current_cost���;�4� +       ��K	��c��A�*

logging/current_cost���;���+       ��K	��c��A�*

logging/current_cost��;Ӆ�{+       ��K	p(d��A�*

logging/current_cost%��;��+       ��K	?Wd��A�*

logging/current_costR��;�0��+       ��K	��d��A�*

logging/current_costd��;/�a+       ��K	��d��A�*

logging/current_cost���;*cma+       ��K	��d��A�*

logging/current_costk��;�/�+       ��K	�e��A�*

logging/current_costN��;#T��+       ��K	z<e��A�*

logging/current_cost���;��96+       ��K	�ie��A�*

logging/current_costu��;�YF�+       ��K	p�e��A�*

logging/current_cost���;B8Q{+       ��K	��e��A�*

logging/current_cost���;uZӎ+       ��K	��e��A�*

logging/current_cost��;���z+       ��K	�!f��A�*

logging/current_cost���;լcb+       ��K	5Of��A�*

logging/current_cost9��;�$��+       ��K	R}f��A�*

logging/current_cost���;��W�+       ��K	,�f��A�*

logging/current_cost���;ע�m+       ��K	v�f��A�*

logging/current_cost��;���+       ��K	:g��A�*

logging/current_cost(�;J�+       ��K	h;g��A�*

logging/current_cost�Q�;_zV+       ��K	Pgg��A�*

logging/current_cost�w�;Crj�+       ��K	�g��A�*

logging/current_cost���;3@+       ��K	��g��A�*

logging/current_costB��;���+       ��K	�h��A�*

logging/current_cost ��;���+       ��K	83h��A�*

logging/current_cost��;�#�1+       ��K	�`h��A�*

logging/current_cost��;+�Ɯ+       ��K	Όh��A�*

logging/current_cost�<�; !Z+       ��K	�h��A�*

logging/current_cost"^�;̑o+       ��K	��h��A�*

logging/current_cost�k�;�2�+       ��K	�i��A�*

logging/current_cost`��;��-+       ��K	Hi��A�*

logging/current_cost��;W�Z�+       ��K	cvi��A�*

logging/current_cost���;��B+       ��K	�i��A�*

logging/current_cost���;��*�+       ��K	��i��A�*

logging/current_cost���;7�Ϟ+       ��K	Lj��A�*

logging/current_cost��;5^!�+       ��K	
0j��A�*

logging/current_cost�;���+       ��K	}^j��A�*

logging/current_cost�#�;��+       ��K	O�j��A�*

logging/current_cost�?�;7f+       ��K	%�j��A�*

logging/current_cost�E�;��K1+       ��K	��j��A�*

logging/current_costrG�;�Xf�+       ��K	mk��A�*

logging/current_cost�_�;����+       ��K	�Fk��A�*

logging/current_cost%Z�;���E+       ��K	)xk��A�*

logging/current_costt�;^��+       ��K	��k��A�*

logging/current_cost��;l��+       ��K	8�k��A�*

logging/current_cost��;���+       ��K	Cl��A�	*

logging/current_costg��;����+       ��K	�6l��A�	*

logging/current_cost�{�;��+       ��K	�il��A�	*

logging/current_cost�p�;�q��+       ��K	Зl��A�	*

logging/current_cost���;��3+       ��K	��l��A�	*

logging/current_cost"z�;j���+       ��K	��l��A�	*

logging/current_cost�;cJT+       ��K	�$m��A�	*

logging/current_cost`��;zJ�r+       ��K	 Sm��A�	*

logging/current_cost��;L璤+       ��K	��m��A�	*

logging/current_cost)��;+���+       ��K	_�m��A�	*

logging/current_cost���;w�h�+       ��K	I�m��A�	*

logging/current_cost���;wg\�+       ��K	Cn��A�	*

logging/current_cost���;�
�R+       ��K	?Cn��A�	*

logging/current_cost���;5C�;+       ��K	Mrn��A�	*

logging/current_cost��;=K��+       ��K	u�n��A�	*

logging/current_costՆ�;��+       ��K	��n��A�	*

logging/current_cost���;�%o&+       ��K	��n��A�	*

logging/current_cost���;�z�++       ��K	�,o��A�	*

logging/current_cost��;/ �+       ��K	�]o��A�	*

logging/current_cost���;���+       ��K	�o��A�	*

logging/current_cost^��;X�+       ��K	��o��A�	*

logging/current_cost��;�8�+       ��K	��o��A�	*

logging/current_costU��;��?+       ��K	�p��A�	*

logging/current_cost��;�_�+       ��K	pFp��A�	*

logging/current_cost���;�b��+       ��K	
vp��A�	*

logging/current_cost��;��\+       ��K	��p��A�
*

logging/current_cost~��;���+       ��K	b�p��A�
*

logging/current_cost��;V��+       ��K	+�p��A�
*

logging/current_cost�p�;jM|�+       ��K	�-q��A�
*

logging/current_cost<i�;g�[+       ��K	[q��A�
*

logging/current_costU`�;�X�+       ��K	��q��A�
*

logging/current_costK]�;߭�K+       ��K	��q��A�
*

logging/current_cost�[�;��-*+       ��K	��q��A�
*

logging/current_cost�[�;v��X+       ��K	r��A�
*

logging/current_cost'V�;tC�+       ��K	�@r��A�
*

logging/current_cost�N�;N��+       ��K	Unr��A�
*

logging/current_cost�@�;h8�+       ��K	/�r��A�
*

logging/current_cost'/�;��S�+       ��K	��r��A�
*

logging/current_cost51�;��+       ��K	��r��A�
*

logging/current_cost'�;�u�+       ��K	S$s��A�
*

logging/current_cost&�;�Lw�+       ��K	�Rs��A�
*

logging/current_cost�!�;�{�+       ��K	̀s��A�
*

logging/current_cost� �;��m�+       ��K	�s��A�
*

logging/current_cost��;h���+       ��K	E�s��A�
*

logging/current_cost��;WOp+       ��K	�t��A�
*

logging/current_cost��;Ԩ�U+       ��K	%>t��A�
*

logging/current_cost��;z��D+       ��K	nt��A�
*

logging/current_cost��;fy+       ��K	]�t��A�
*

logging/current_cost4�;� �+       ��K	"�t��A�
*

logging/current_cost0�;G�+       ��K	Cu��A�
*

logging/current_cost��;�N��+       ��K	�/u��A�
*

logging/current_cost��;���+       ��K	�_u��A�*

logging/current_costr�;+RA+       ��K	�u��A�*

logging/current_costP�;XqI�+       ��K	�u��A�*

logging/current_cost��;_n�+       ��K	��u��A�*

logging/current_costY�;S=�V+       ��K	�v��A�*

logging/current_cost��;��r�+       ��K	Bv��A�*

logging/current_cost^(�;���+       ��K	�qv��A�*

logging/current_costu+�;�֌+       ��K	��v��A�*

logging/current_costY,�;wFL�+       ��K	=�v��A�*

logging/current_cost�/�;�>�+       ��K	��v��A�*

logging/current_cost<5�;�q��+       ��K	+w��A�*

logging/current_cost�9�;�䨭+       ��K	TXw��A�*

logging/current_costu<�;rY+       ��K	�w��A�*

logging/current_costU>�;t ��+       ��K	��w��A�*

logging/current_costNA�;����+       ��K	��w��A�*

logging/current_cost�C�;��tx+       ��K	�x��A�*

logging/current_cost�?�;��K�+       ��K	�Ex��A�*

logging/current_cost�>�;�(��+       ��K	.rx��A�*

logging/current_cost�?�;�^	�+       ��K	��x��A�*

logging/current_cost�@�;�9߰+       ��K	�x��A�*

logging/current_cost�<�;���h+       ��K	�y��A�*

logging/current_cost'8�;nc��+       ��K	$.y��A�*

logging/current_cost�1�;�k	�+       ��K	,_y��A�*

logging/current_costu0�;����+       ��K	3�y��A�*

logging/current_cost4-�;��ƍ+       ��K	v�y��A�*

logging/current_cost{)�;O[��+       ��K	��y��A�*

logging/current_cost&�;��`+       ��K	�z��A�*

logging/current_costY �;�-O�+       ��K	�Ez��A�*

logging/current_cost��;�@+       ��K	�sz��A�*

logging/current_cost�"�;�B�+       ��K	��z��A�*

logging/current_cost�!�;wC�l+       ��K	��z��A�*

logging/current_cost"�;�ڍ5+       ��K	�{��A�*

logging/current_cost<'�;���P+       ��K	�2{��A�*

logging/current_cost5,�;Z.��+       ��K	Ca{��A�*

logging/current_cost�,�;t=��+       ��K	?�{��A�*

logging/current_coste,�;5��+       ��K	�{��A�*

logging/current_cost@-�;ac�+       ��K	�1|��A�*

logging/current_costy2�;�U'B+       ��K	*i|��A�*

logging/current_cost�4�;7�+       ��K	ȡ|��A�*

logging/current_cost�4�;Z�UM+       ��K	��|��A�*

logging/current_cost�7�;��y+       ��K	�}��A�*

logging/current_cost�;�;��e+       ��K	fT}��A�*

logging/current_cost�=�;y!�+       ��K	��}��A�*

logging/current_cost�?�;W=��+       ��K	��}��A�*

logging/current_cost�A�;eN#�+       ��K	�
~��A�*

logging/current_cost�D�;�o��+       ��K	�A~��A�*

logging/current_costA�;M
t�+       ��K	sq~��A�*

logging/current_cost[F�;��+       ��K	�~��A�*

logging/current_cost2R�;tKs�+       ��K	�~��A�*

logging/current_cost�V�;ͤ�5+       ��K	{��A�*

logging/current_cost�U�;#b��+       ��K	�6��A�*

logging/current_cost�V�;9f+       ��K	1h��A�*

logging/current_cost	]�;��(+       ��K	l���A�*

logging/current_costb�;,Ζ+       ��K	Z���A�*

logging/current_cost�d�;�bʚ+       ��K	����A�*

logging/current_cost7g�;@+       ��K	�'���A�*

logging/current_cost�k�;ِ�+       ��K	�V���A�*

logging/current_cost�p�;����+       ��K	_����A�*

logging/current_cost�t�;s�+       ��K	�����A�*

logging/current_costpx�;�B�+       ��K	�瀷�A�*

logging/current_cost�|�;@��Q+       ��K	/���A�*

logging/current_cost^��;�Զ+       ��K	 I���A�*

logging/current_costą�;��B+       ��K	~v���A�*

logging/current_costՉ�;{��+       ��K	{����A�*

logging/current_cost���;ѹ�j+       ��K	Iԁ��A�*

logging/current_cost��;Έ+N+       ��K	i���A�*

logging/current_cost ��;QH�O+       ��K	"9���A�*

logging/current_cost��;�qA+       ��K	o���A�*

logging/current_costΪ�;��U0+       ��K	ǥ���A�*

logging/current_coste��;�,|h+       ��K	Ԃ��A�*

logging/current_cost5��;��Z+       ��K	����A�*

logging/current_cost���;��L�+       ��K	4���A�*

logging/current_cost,��;i=�+       ��K	o���A�*

logging/current_cost4��;a��+       ��K	J����A�*

logging/current_cost���;���+       ��K	ރ��A�*

logging/current_cost���;�ƶl+       ��K	L���A�*

logging/current_cost2��;L�jx+       ��K	F���A�*

logging/current_cost.��;)%��+       ��K	Fw���A�*

logging/current_cost5 �;��6+       ��K	;����A�*

logging/current_cost�	�;SM��+       ��K	�߄��A�*

logging/current_costw�;y��+       ��K	v���A�*

logging/current_cost��;D�+       ��K	�D���A�*

logging/current_cost� �;�$7+       ��K	�t���A�*

logging/current_cost�$�;j��+       ��K	˩���A�*

logging/current_cost�2�;`�`+       ��K	߅��A�*

logging/current_cost�C�;.!3�+       ��K	����A�*

logging/current_costNG�;�Ly�+       ��K	vC���A�*

logging/current_cost<L�;��TQ+       ��K	�v���A�*

logging/current_costW�;�3�+       ��K	�����A�*

logging/current_costdd�;F|�+       ��K	�ㆷ�A�*

logging/current_cost�e�;Ѕu�+       ��K	�*���A�*

logging/current_cost�k�;bs+       ��K	�n���A�*

logging/current_cost�v�;2��7+       ��K	�̇��A�*

logging/current_costˁ�;w<G�+       ��K	����A�*

logging/current_cost5��;\!%+       ��K	�3���A�*

logging/current_cost���;�%��+       ��K	j���A�*

logging/current_costK��;wbu+       ��K	c����A�*

logging/current_cost	��;��E#+       ��K	�Ј��A�*

logging/current_cost"��;�g�t+       ��K	</���A�*

logging/current_cost`��;h��Q+       ��K	Wy���A�*

logging/current_costk��;~~9�+       ��K	�ŉ��A�*

logging/current_cost���;�P!\+       ��K	k ���A�*

logging/current_costk��;��++       ��K	�Y���A�*

logging/current_cost���;MI<�+       ��K	T����A�*

logging/current_cost5��;Jj+       ��K	�����A�*

logging/current_cost���;�/�+       ��K	f,���A�*

logging/current_cost;��;i�$+       ��K	I[���A�*

logging/current_cost���;E�+       ��K	K����A�*

logging/current_cost��;48�+       ��K	.ϋ��A�*

logging/current_cost��;J��+       ��K	����A�*

logging/current_cost��;��+       ��K	�H���A�*

logging/current_cost�
�;/�V+       ��K	~���A�*

logging/current_cost��;�# +       ��K	Ʋ���A�*

logging/current_cost{�;��&Z+       ��K	�挷�A�*

logging/current_costg �;qf�j+       ��K	����A�*

logging/current_cost/�;�@+       ��K	�R���A�*

logging/current_cost7�;=n�+       ��K	�����A�*

logging/current_cost=�;X{d�+       ��K	y����A�*

logging/current_costA�;�y�+       ��K	�퍷�A�*

logging/current_cost�K�;v��+       ��K	d/���A�*

logging/current_costgR�;0�ѫ+       ��K	$k���A�*

logging/current_costTP�;��(5+       ��K	�����A�*

logging/current_costi^�;c^�<+       ��K	bˎ��A�*

logging/current_cost.f�;���>+       ��K	]����A�*

logging/current_cost q�;�8��+       ��K	�-���A�*

logging/current_costx�;<k�~+       ��K	�i���A�*

logging/current_cost���;����+       ��K	W����A�*

logging/current_cost;��; �)+       ��K	�ۏ��A�*

logging/current_cost��;/y�/+       ��K	3���A�*

logging/current_costP��;s0�C+       ��K	9���A�*

logging/current_costΚ�;�?�+       ��K	Qr���A�*

logging/current_cost̤�;�?�`+       ��K	�����A�*

logging/current_cost.��;����+       ��K	�Ґ��A�*

logging/current_cost���;�\�+       ��K	����A�*

logging/current_costK��;�� +       ��K	t5���A�*

logging/current_cost��;.3�n+       ��K	mi���A�*

logging/current_cost���;!α�+       ��K	
����A�*

logging/current_cost[��;��:c+       ��K	̑��A�*

logging/current_cost���;�ò+       ��K	x����A�*

logging/current_cost���;���y+       ��K	�,���A�*

logging/current_costG��;m`�>+       ��K	�\���A�*

logging/current_cost���;�b�_+       ��K	m����A�*

logging/current_cost��;��R+       ��K	JȒ��A�*

logging/current_cost� �;.�r�+       ��K	�����A�*

logging/current_cost�;���*+       ��K	 )���A�*

logging/current_cost��;���Y+       ��K	�X���A�*

logging/current_cost�*�;����+       ��K	@����A�*

logging/current_cost�=�;�	k�+       ��K	���A�*

logging/current_cost�L�;�l��+       ��K	�����A�*

logging/current_cost|`�;*�+       ��K	�.���A�*

logging/current_cost�s�;�`�+       ��K	�^���A�*

logging/current_cost.��;�lp�+       ��K	�����A�*

logging/current_costD��;��j-+       ��K	�Δ��A�*

logging/current_cost{��;�{�<+       ��K	o����A�*

logging/current_cost��;P�M�+       ��K	80���A�*

logging/current_costg��;���=+       ��K	�`���A�*

logging/current_cost��;��$+       ��K	E����A�*

logging/current_cost��;gA�+       ��K	�ѕ��A�*

logging/current_cost�;U���+       ��K	����A�*

logging/current_costk-�;s���+       ��K	�7���A�*

logging/current_cost�E�;Ni�+       ��K	�h���A�*

logging/current_cost�U�;/cS+       ��K	����A�*

logging/current_cost�j�;�n'�+       ��K	�͖��A�*

logging/current_cost�y�;E_�(+       ��K	e����A�*

logging/current_cost��;*`;�+       ��K	�-���A�*

logging/current_costG��;��,f+       ��K	�[���A�*

logging/current_cost��;� �+       ��K	a����A�*

logging/current_cost{��;"}��+       ��K	Ǻ���A�*

logging/current_cost���;pdON+       ��K	]��A�*

logging/current_cost���;3�2+       ��K	�!���A�*

logging/current_cost��;@�+       ��K	{O���A�*

logging/current_costT
�;�}Ȉ+       ��K	g|���A�*

logging/current_cost"�;G��O+       ��K	I����A�*

logging/current_cost�3�;��#�+       ��K	Zؘ��A�*

logging/current_cost|?�;���+       ��K	�	���A�*

logging/current_cost�O�;�Rj+       ��K	s8���A�*

logging/current_cost�f�;J+       ��K	�h���A�*

logging/current_cost^z�;ys'j+       ��K	>����A�*

logging/current_cost@��;�z�+       ��K	yř��A�*

logging/current_costy��;� \4+       ��K	���A�*

logging/current_cost���;*U�+       ��K	�%���A�*

logging/current_cost���;��9+       ��K	�Y���A�*

logging/current_cost���;���+       ��K	�����A�*

logging/current_costw��;��j�+       ��K	�����A�*

logging/current_cost0��;���+       ��K	�ꚷ�A�*

logging/current_costD��;�_��+       ��K	���A�*

logging/current_cost`�;��E6+       ��K	hJ���A�*

logging/current_cost� �;���+       ��K		{���A�*

logging/current_cost92�;ʐgE+       ��K	l����A�*

logging/current_cost�@�;B
�4+       ��K	�֛��A�*

logging/current_costgO�;��]+       ��K	�	���A�*

logging/current_cost�_�;�G�Q+       ��K	8:���A�*

logging/current_costp�;���k+       ��K	#i���A�*

logging/current_cost܀�;�L>�+       ��K	g����A�*

logging/current_cost\��;A*�R+       ��K	sŜ��A�*

logging/current_costբ�;�S�l+       ��K	�����A�*

logging/current_cost��;�|z+       ��K	�"���A�*

logging/current_costU��;�ZS�+       ��K	�R���A�*

logging/current_cost���;v	'�+       ��K	�����A�*

logging/current_cost���;m���+       ��K	�����A�*

logging/current_cost���;p.ȗ+       ��K	�杷�A�*

logging/current_cost���;'���+       ��K	����A�*

logging/current_cost���;un�=+       ��K	]E���A�*

logging/current_cost��;�?�+       ��K	Ns���A�*

logging/current_costb!�;Mَ�+       ��K	�����A�*

logging/current_cost)3�;H	�.+       ��K	$͞��A�*

logging/current_cost�7�;V��+       ��K	����A�*

logging/current_cost�G�;V��f+       ��K	�,���A�*

logging/current_cost�W�;@��+       ��K	�X���A�*

logging/current_cost	o�;L,
�+       ��K	�����A�*

logging/current_costǍ�;'��+       ��K	�����A�*

logging/current_costd��;	��+       ��K	*執�A�*

logging/current_cost���;n��+       ��K	$���A�*

logging/current_cost���;0�(X+       ��K	B���A�*

logging/current_cost5��;쯯Z+       ��K	�p���A�*

logging/current_cost%��;!��+       ��K	ݟ���A�*

logging/current_cost���;և��+       ��K	WΠ��A�*

logging/current_cost��;>/a.+       ��K	����A�*

logging/current_cost�.�;�v'+       ��K	�*���A�*

logging/current_cost�D�;���&+       ��K	=W���A�*

logging/current_cost�^�;����+       ��K	놡��A�*

logging/current_costk�;��}+       ��K	T����A�*

logging/current_cost}�;�J=+       ��K	�桷�A�*

logging/current_cost'��;�k+       ��K	���A�*

logging/current_cost��;�5#c+       ��K	�B���A�*

logging/current_costn��;�t�+       ��K	8s���A�*

logging/current_cost���;�$\3+       ��K	����A�*

logging/current_costp��;[bg+       ��K	�Т��A�*

logging/current_cost2f�;x%G_+       ��K	T����A�*

logging/current_cost�T�;��M+       ��K	r,���A�*

logging/current_costo�;��!#+       ��K	�\���A�*

logging/current_cost�V�;�m�+       ��K	2����A�*

logging/current_cost9X�;WsU+       ��K	}����A�*

logging/current_cost�G�;Õ +       ��K	�磷�A�*

logging/current_cost�B�;ь�\+       ��K	����A�*

logging/current_cost�8�;y@��+       ��K	�C���A�*

logging/current_costr5�;��۳+       ��K	�s���A�*

logging/current_cost�4�;�)�+       ��K	"����A�*

logging/current_cost�4�;!�+       ��K	
Ф��A�*

logging/current_cost�9�;@�DY+       ��K	�����A�*

logging/current_cost:�;k(ht+       ��K	�.���A�*

logging/current_costE;�;sR�+       ��K	H`���A�*

logging/current_cost�9�;{��+       ��K	�����A�*

logging/current_cost�<�;	�,+       ��K	�����A�*

logging/current_cost�<�;�y�^+       ��K	�楷�A�*

logging/current_cost�<�;�n�+       ��K	����A�*

logging/current_cost\@�;90�+       ��K	�@���A�*

logging/current_cost\E�;C�+       ��K	�p���A�*

logging/current_cost�D�;��`+       ��K	���A�*

logging/current_cost�J�;J�+       ��K	&˦��A�*

logging/current_costNF�;�R��+       ��K	;����A�*

logging/current_costUI�;7��H+       ��K	A-���A�*

logging/current_cost2M�;�c�e+       ��K	e[���A�*

logging/current_cost\J�;�ܙ�+       ��K	�����A�*

logging/current_cost�K�;+�+       ��K	�����A�*

logging/current_costG�;h6+       ��K	�觷�A�*

logging/current_costyE�;�63+       ��K	����A�*

logging/current_costK�;0�+       ��K	�C���A�*

logging/current_cost�K�;t�X+       ��K	q���A�*

logging/current_cost�L�;^�/+       ��K	柨��A�*

logging/current_costWJ�;;o+       ��K	Ψ��A�*

logging/current_costK�;~9}�+       ��K	`���A�*

logging/current_cost�S�;7x0�+       ��K	*2���A�*

logging/current_cost~N�;T�|h+       ��K	�_���A�*

logging/current_costIK�;�g+       ��K	j����A�*

logging/current_cost�V�;����+       ��K	�����A�*

logging/current_cost�N�;�п�+       ��K	��A�*

logging/current_cost�X�;��A�+       ��K	����A�*

logging/current_cost�N�;doh�+       ��K	&J���A�*

logging/current_cost�[�;���a+       ��K	x���A�*

logging/current_cost<N�;�-ļ+       ��K	�����A�*

logging/current_cost�X�;��4P+       ��K	�ժ��A�*

logging/current_cost�V�;�8h+       ��K	����A�*

logging/current_cost�T�;��c�+       ��K	�:���A�*

logging/current_costRS�;Hf��+       ��K	�h���A�*

logging/current_cost.S�;�(b�+       ��K	{����A�*

logging/current_costkR�;�x�+       ��K	~����A�*

logging/current_costTT�;}(��+       ��K	N����A�*

logging/current_cost�S�;�y�^+       ��K	�"���A�*

logging/current_cost�\�;0x��+       ��K	YR���A�*

logging/current_cost�V�;(�a�+       ��K	����A�*

logging/current_costwU�;���+       ��K	ڰ���A�*

logging/current_cost"\�;����+       ��K	�ଷ�A�*

logging/current_cost�U�;]�E+       ��K	s���A�*

logging/current_cost�]�;�N/�+       ��K	>���A�*

logging/current_cost�V�;���+       ��K	�n���A�*

logging/current_costa�;)�Z+       ��K	˜���A�*

logging/current_cost�\�;�+       ��K	�ȭ��A�*

logging/current_cost�c�;����+       ��K	(����A�*

logging/current_cost�a�;2�o+       ��K	L)���A�*

logging/current_costeb�; ���+       ��K	�W���A�*

logging/current_cost�a�;|�'d+       ��K	V����A�*

logging/current_costYj�;�"�r+       ��K	8����A�*

logging/current_cost�j�;m)�+       ��K		㮷�A�*

logging/current_cost2i�;u;�+       ��K	����A�*

logging/current_cost�m�;1@�y+       ��K	�B���A�*

logging/current_cost<i�;((�+       ��K	{p���A�*

logging/current_cost�m�;&{�k+       ��K	읯��A�*

logging/current_cost�j�;fL:�+       ��K	ɯ��A�*

logging/current_cost�m�;�{�+       ��K	�����A�*

logging/current_cost�w�;��8+       ��K	[$���A�*

logging/current_cost	q�;���V+       ��K	R���A�*

logging/current_cost�|�;q��+       ��K	�����A�*

logging/current_cost���;��+       ��K	�����A�*

logging/current_cost��;b�"�+       ��K	�۰��A�*

logging/current_cost���;�KI�+       ��K	����A�*

logging/current_cost��;
*|+       ��K	�6���A�*

logging/current_cost��;=�!-+       ��K	Gb���A�*

logging/current_cost���;�W|5+       ��K	呱��A�*

logging/current_cost%��;~4V�+       ��K	>����A�*

logging/current_costr��;�;	�+       ��K	�����A�*

logging/current_cost���;'Պ+       ��K	���A�*

logging/current_costk��;
�+�+       ��K	6J���A�*

logging/current_cost�m�;hWv+       ��K	�u���A�*

logging/current_costwd�;�M�+       ��K	`����A�*

logging/current_cost�K�;����+       ��K	�в��A�*

logging/current_costE�;�M�L+       ��K	�����A�*

logging/current_cost�9�;�.Zn+       ��K	-���A�*

logging/current_cost�"�;e�{�+       ��K	dZ���A�*

logging/current_costU)�;�I-+       ��K	#����A�*

logging/current_cost�;|-�v+       ��K	U����A�*

logging/current_cost��;e��U+       ��K	%泷�A�*

logging/current_cost"
�;J�!+       ��K	����A�*

logging/current_cost��;1�ذ+       ��K	�B���A�*

logging/current_cost��;V��u+       ��K	s���A�*

logging/current_costk
�;�K��+       ��K	 ���A�*

logging/current_cost���;�͹+       ��K	ϴ��A�*

logging/current_cost�;�:3+       ��K	�����A�*

logging/current_cost��;G�D�+       ��K	�)���A�*

logging/current_cost���;��L�+       ��K	�W���A�*

logging/current_cost�	�;v��E+       ��K	v����A�*

logging/current_cost���;�,·+       ��K	�����A�*

logging/current_cost��;� �6+       ��K	V⵷�A�*

logging/current_cost���;ο+       ��K	P���A�*

logging/current_cost��;w�T�+       ��K	5C���A�*

logging/current_cost �;�X�1+       ��K	�r���A�*

logging/current_cost��;$���+       ��K	6����A�*

logging/current_cost~�;`���+       ��K	�̶��A�*

logging/current_cost��;��W�+       ��K	7����A�*

logging/current_cost5�;���+       ��K	�)���A�*

logging/current_cost^�;Ƒ�d+       ��K	W���A�*

logging/current_cost'"�;O~y+       ��K	釷��A�*

logging/current_cost~�;�N:+       ��K	�����A�*

logging/current_cost��;�}u^+       ��K	緷�A�*

logging/current_cost%�;#S-'+       ��K	����A�*

logging/current_cost�	�;.�4+       ��K	�E���A�*

logging/current_cost��;���W+       ��K	�t���A�*

logging/current_cost[�;�_+       ��K	U����A�*

logging/current_cost�;}�+       ��K	]и��A�*

logging/current_cost��;��Z�+       ��K	����A�*

logging/current_costN�;cy�+       ��K	X/���A�*

logging/current_cost0�;&��2+       ��K	d]���A�*

logging/current_cost��;�1	q+       ��K	�����A�*

logging/current_cost��;ca��+       ��K	b����A�*

logging/current_cost��;�=��+       ��K	�湷�A�*

logging/current_cost�;ݴ�++       ��K	���A�*

logging/current_cost��;�݂A+       ��K	�?���A�*

logging/current_cost�;�͐�+       ��K	-r���A�*

logging/current_cost$�;V���+       ��K	;����A�*

logging/current_cost\�;3��+       ��K	�κ��A�*

logging/current_cost��;&�F�+       ��K	�����A�*

logging/current_cost,�;�2k�+       ��K	/.���A�*

logging/current_costW�;�]�+       ��K	}]���A�*

logging/current_cost��;��|	+       ��K	����A�*

logging/current_cost��;�uM+       ��K	|����A�*

logging/current_costW�;\��J+       ��K	4���A�*

logging/current_cost��;���++       ��K	jn���A�*

logging/current_costY�;#�Tn+       ��K	M����A�*

logging/current_cost�;���U+       ��K	"�A�*

logging/current_cost%#�;A���+       ��K	�%���A�*

logging/current_costI!�;��d+       ��K	�^���A�*

logging/current_cost�"�;�ܓ~+       ��K	�����A�*

logging/current_cost "�;��R�+       ��K	Խ��A�*

logging/current_cost� �;b��+       ��K	�
���A�*

logging/current_cost2!�;q �]+       ��K	�;���A�*

logging/current_cost+)�;'E��+       ��K	�}���A�*

logging/current_costn �;�U�k+       ��K	3����A�*

logging/current_cost|(�;p�l�+       ��K	3⾷�A�*

logging/current_cost�;c��8+       ��K	����A�*

logging/current_cost�)�;r���+       ��K	�B���A�*

logging/current_cost��;n�X+       ��K	lq���A�*

logging/current_cost�.�;u�ye+       ��K	$����A�*

logging/current_cost�;S���+       ��K	Qп��A�*

logging/current_cost�+�;$�O+       ��K	b���A�*

logging/current_cost>'�;���:+       ��K	91���A�*

logging/current_cost� �;K�;+       ��K	^���A�*

logging/current_cost�/�;����+       ��K	����A�*

logging/current_cost��;�c�+       ��K	����A�*

logging/current_costp3�;�A�~+       ��K	�����A�*

logging/current_cost�#�;(#ǟ+       ��K	 ���A�*

logging/current_cost�#�;s��c+       ��K	�J���A�*

logging/current_cost�)�;:Sg+       ��K	Pz���A�*

logging/current_cost�)�;q��W+       ��K	7����A�*

logging/current_cost )�;��3+       ��K	����A�*

logging/current_cost))�;%$<L+       ��K	�·�A�*

logging/current_cost�2�;��+       ��K	^7·�A�*

logging/current_cost�&�;Q�?�+       ��K	+l·�A�*

logging/current_cost�0�;�<w�+       ��K	3�·�A�*

logging/current_costk&�;��+       ��K	j�·�A�*

logging/current_cost�.�;]�ٻ+       ��K	�q÷�A�*

logging/current_cost�-�;�ŞI+       ��K	԰÷�A�*

logging/current_costD'�;%K+       ��K	��÷�A�*

logging/current_cost�.�;4���+       ��K	tķ�A�*

logging/current_cost{#�;y��-+       ��K	9Sķ�A�*

logging/current_cost�)�;�� �+       ��K	��ķ�A�*

logging/current_cost(�;ִ_7+       ��K	x�ķ�A�*

logging/current_cost.,�;��C�+       ��K	ŷ�A�*

logging/current_cost��;"�
5+       ��K	Hŷ�A�*

logging/current_cost�$�;M��+       ��K	T�ŷ�A�*

logging/current_cost�#�; �c�+       ��K	�ŷ�A�*

logging/current_cost*�;1m1+       ��K	r�ŷ�A�*

logging/current_cost��;�n{+       ��K	/Ʒ�A�*

logging/current_cost!�;��ޙ+       ��K	�SƷ�A�*

logging/current_cost�(�;sߛ>+       ��K	d�Ʒ�A�*

logging/current_cost��;���~+       ��K	�Ʒ�A�*

logging/current_cost�*�;]�/�+       ��K	��Ʒ�A�*

logging/current_costb�;�8v(+       ��K	�,Ƿ�A�*

logging/current_cost~)�;^���+       ��K	aaǷ�A�*

logging/current_cost��;��D+       ��K	ޠǷ�A�*

logging/current_cost )�;����+       ��K	��Ƿ�A�*

logging/current_costW%�;�'1+       ��K	}	ȷ�A�*

logging/current_costu�;N�N=+       ��K	�8ȷ�A�*

logging/current_cost�.�;6�P+       ��K	lkȷ�A�*

logging/current_costT�;~���+       ��K	��ȷ�A�*

logging/current_cost�0�;��6�+       ��K	��ȷ�A�*

logging/current_cost��;��+       ��K	'ɷ�A�*

logging/current_cost,*�;`So+       ��K	!@ɷ�A�*

logging/current_costu&�;ĩ�t+       ��K	;{ɷ�A�*

logging/current_cost��;i��G+       ��K	��ɷ�A�*

logging/current_cost�2�;�:�$+       ��K	��ɷ�A�*

logging/current_cost��;��rl+       ��K	
ʷ�A�*

logging/current_cost�,�;��
+       ��K	�:ʷ�A�*

logging/current_cost�%�;,}(�+       ��K	lʷ�A�*

logging/current_cost�-�;xv��+       ��K	*�ʷ�A�*

logging/current_cost2�;��k�+       ��K	��ʷ�A�*

logging/current_cost�*�;�A�+       ��K	�ʷ�A�*

logging/current_cost�4�;��+       ��K	�$˷�A�*

logging/current_cost|(�;�ݱ�+       ��K	�R˷�A�*

logging/current_cost�*�;�0�+       ��K	�˷�A�*

logging/current_cost�/�;��p�+       ��K	;�˷�A�*

logging/current_cost�-�;�zR�+       ��K	��˷�A�*

logging/current_cost�2�;�[�+       ��K	�̷�A�*

logging/current_costY,�;��S�+       ��K	cM̷�A�*

logging/current_costB9�;�	�b+       ��K	�|̷�A�*

logging/current_cost&�;���A+       ��K	2�̷�A�*

logging/current_costy;�;$���+       ��K	�̷�A�*

logging/current_cost�6�;�v�+       ��K	�
ͷ�A�*

logging/current_cost�5�;I*i�+       ��K	�6ͷ�A�*

logging/current_cost�/�;��+       ��K	0gͷ�A�*

logging/current_costp5�;���+       ��K	��ͷ�A�*

logging/current_cost.=�;�5�+       ��K	��ͷ�A�*

logging/current_cost�'�;�׮R+       ��K	��ͷ�A�*

logging/current_cost[C�;l�^+       ��K	q&η�A�*

logging/current_cost�/�;I��i+       ��K	�\η�A�*

logging/current_cost5<�;����+       ��K	Ћη�A�*

logging/current_cost�:�;���+       ��K	��η�A�*

logging/current_costb<�;[4+       ��K	��η�A�*

logging/current_cost�>�;���(+       ��K	Ϸ�A�*

logging/current_cost�=�;��O+       ��K	�HϷ�A�*

logging/current_costp>�;�7�\+       ��K	.�Ϸ�A�*

logging/current_costD�;s��P+       ��K	K�Ϸ�A�*

logging/current_costU:�;�P��+       ��K	F�Ϸ�A�*

logging/current_cost�;�;��+       ��K	 "з�A�*

logging/current_cost�D�;��+       ��K	�Oз�A�*

logging/current_cost	>�;�7%�+       ��K	�з�A�*

logging/current_cost,7�;��+       ��K	B�з�A�*

logging/current_cost�A�;�`�,+       ��K	��з�A�*

logging/current_cost`G�;�A��+       ��K	�ѷ�A�*

logging/current_costyA�;޳�+       ��K	�@ѷ�A�*

logging/current_costF�;;[+       ��K	�pѷ�A�*

logging/current_cost�B�;�c�}+       ��K	�ѷ�A�*

logging/current_cost�C�;҃+       ��K	�ѷ�A�*

logging/current_cost�N�;[�f�+       ��K	� ҷ�A�*

logging/current_cost5<�;�B�+       ��K	�2ҷ�A�*

logging/current_cost�C�;�v��+       ��K	"gҷ�A�*

logging/current_cost�J�;�aF�+       ��K	�ҷ�A�*

logging/current_cost�H�;F���+       ��K	��ҷ�A�*

logging/current_cost�E�;��a+       ��K	�ӷ�A�*

logging/current_cost2D�;D���+       ��K	�Nӷ�A�*

logging/current_costuS�;ͣ��+       ��K	p|ӷ�A�*

logging/current_cost?�;��8�+       ��K	�ӷ�A�*

logging/current_costUV�;��mK+       ��K	��ӷ�A�*

logging/current_cost�B�;�ųV+       ��K	� Է�A�*

logging/current_cost�F�;lP,�+       ��K	�TԷ�A�*

logging/current_cost^I�;X�`|+       ��K	�Է�A�*

logging/current_costLU�;K�,+       ��K	��Է�A�*

logging/current_cost[B�;�G�R+       ��K	A�Է�A�*

logging/current_cost�I�;��N+       ��K	gշ�A�*

logging/current_costM�;D�j+       ��K	�Nշ�A�*

logging/current_cost~G�;����+       ��K	M~շ�A�*

logging/current_cost\M�;pNu+       ��K	��շ�A�*

logging/current_costU�;��[+       ��K	��շ�A�*

logging/current_cost<J�;fK<O+       ��K	)ַ�A�*

logging/current_costwN�;�X�+       ��K	�Jַ�A�*

logging/current_cost�Q�;b/0+       ��K	L|ַ�A�*

logging/current_cost4T�;8��+       ��K	|�ַ�A�*

logging/current_costKV�;��۷+       ��K	��ַ�A�*

logging/current_costBZ�;���z+       ��K	�׷�A�*

logging/current_cost�D�;�zK�+       ��K	=W׷�A�*

logging/current_cost|P�;���G+       ��K	��׷�A�*

logging/current_cost	Q�;�+       ��K	�׷�A�*

logging/current_cost$S�;�,�+       ��K	��׷�A�*

logging/current_cost%S�;n.�+       ��K	�ط�A�*

logging/current_costiO�;$CMY+       ��K	OLط�A�*

logging/current_cost>R�;-=��+       ��K	b~ط�A�*

logging/current_costrX�;GL+8+       ��K	��ط�A�*

logging/current_costtQ�;\#�I+       ��K	A�ط�A�*

logging/current_costM�;�WZ+       ��K	�ٷ�A�*

logging/current_cost�X�;ğm�+       ��K	nNٷ�A�*

logging/current_costW�;��@}+       ��K	�ٷ�A�*

logging/current_cost�`�;=&�_+       ��K	��ٷ�A�*

logging/current_cost�M�;�@L�+       ��K	��ٷ�A�*

logging/current_cost�M�;k/v�+       ��K	�
ڷ�A�*

logging/current_cost�_�;��҅+       ��K	n?ڷ�A�*

logging/current_cost�T�;��Ǎ+       ��K	[pڷ�A�*

logging/current_costYe�;��*�+       ��K	ӟڷ�A�*

logging/current_cost�V�;�B�=+       ��K	�ڷ�A�*

logging/current_cost�N�;x"��+       ��K	��ڷ�A�*

logging/current_costnX�;pju?+       ��K	�.۷�A�*

logging/current_cost�Y�;��7�+       ��K	Zd۷�A�*

logging/current_costY_�;��+       ��K	@�۷�A�*

logging/current_costdT�;q���+       ��K	��۷�A�*

logging/current_cost�Y�;��'+       ��K	��۷�A�*

logging/current_cost\Z�;���+       ��K	�ܷ�A�*

logging/current_cost�^�;0���+       ��K	Oܷ�A�*

logging/current_cost{]�;�Bb�+       ��K	�ܷ�A�*

logging/current_cost�]�;�z@�+       ��K	�ܷ�A�*

logging/current_cost�J�;ҟ�p+       ��K	��ܷ�A�*

logging/current_cost�\�;ߏ��+       ��K	�ݷ�A�*

logging/current_cost�X�;�P?+       ��K	�Mݷ�A�*

logging/current_costWn�;���+       ��K	]ݷ�A�*

logging/current_costWQ�;X���+       ��K	R�ݷ�A�*

logging/current_cost�T�;�Z�W+       ��K	��ݷ�A�*

logging/current_cost�b�;�u�+       ��K	h޷�A�*

logging/current_cost�Z�;\���+       ��K	Y;޷�A�*

logging/current_costgg�;��1)+       ��K	in޷�A�*

logging/current_cost<]�;B�+       ��K	��޷�A�*

logging/current_cost�P�;�쁤+       ��K	��޷�A�*

logging/current_cost�h�;�f�+       ��K	��޷�A� *

logging/current_cost�O�;�R�+       ��K	T.߷�A� *

logging/current_cost�c�;d`�~+       ��K	�b߷�A� *

logging/current_cost i�;M9�S+       ��K	p�߷�A� *

logging/current_cost+S�;�H.+       ��K	s�߷�A� *

logging/current_cost@g�;�TȒ+       ��K	���A� *

logging/current_cost	[�;���+       ��K	�1��A� *

logging/current_cost�\�;���+       ��K	�`��A� *

logging/current_cost�^�;�-�+       ��K	���A� *

logging/current_cost�`�;r��+       ��K	����A� *

logging/current_costc�;b8A�+       ��K	0���A� *

logging/current_costyi�;���+       ��K	"1��A� *

logging/current_costbY�;�d��+       ��K	7`��A� *

logging/current_costiY�;��B+       ��K	D���A� *

logging/current_cost�g�;i>�K+       ��K	4���A� *

logging/current_cost ]�;s��b+       ��K	.���A� *

logging/current_cost ^�;!ň+       ��K	X"��A� *

logging/current_cost@V�;HGM�+       ��K	�T��A� *

logging/current_cost\e�;Ź+       ��K	���A� *

logging/current_cost�d�;��+       ��K	����A� *

logging/current_cost$c�;/f��+       ��K	8���A� *

logging/current_cost�d�;���
+       ��K	S��A� *

logging/current_costP�;0�A;+       ��K	�E��A� *

logging/current_cost�f�;a	�i+       ��K	�t��A� *

logging/current_costa�;>�Fj+       ��K	B���A� *

logging/current_cost_�; ]r+       ��K	����A�!*

logging/current_costbj�;�EA'+       ��K	����A�!*

logging/current_cost�a�;�A�z+       ��K	O-��A�!*

logging/current_cost e�;����+       ��K	G]��A�!*

logging/current_cost|c�;��X
+       ��K	ċ��A�!*

logging/current_cost�d�;TD�+       ��K	3���A�!*

logging/current_cost�R�;;�L�+       ��K	Q���A�!*

logging/current_costg�;}a7 +       ��K		��A�!*

logging/current_cost�d�;׮��+       ��K	 B��A�!*

logging/current_cost$f�;���+       ��K	�o��A�!*

logging/current_cost�b�;K��+       ��K	b���A�!*

logging/current_cost�d�;���+       ��K	����A�!*

logging/current_cost.[�;�=m+       ��K	!���A�!*

logging/current_cost�c�;/�+       ��K	�'��A�!*

logging/current_cost�o�;�g��+       ��K	$W��A�!*

logging/current_cost�W�;T���+       ��K	 ���A�!*

logging/current_costRk�;��G�+       ��K	����A�!*

logging/current_cost�X�;��k+       ��K	���A�!*

logging/current_cost�m�;�Z�r+       ��K	gA��A�!*

logging/current_cost�S�;C��m+       ��K	-o��A�!*

logging/current_cost.Q�;8�a+       ��K	ɞ��A�!*

logging/current_cost�q�;� &�+       ��K	2���A�!*

logging/current_cost�U�;7G��+       ��K	���A�!*

logging/current_cost^d�;a��q+       ��K	�1��A�!*

logging/current_cost�e�;���8+       ��K	�`��A�!*

logging/current_cost;h�;�x� +       ��K	����A�!*

logging/current_cost�e�;�d��+       ��K	����A�"*

logging/current_cost�p�;�̉+       ��K	W���A�"*

logging/current_cost�Y�;��X�+       ��K	�%��A�"*

logging/current_cost|c�;��Ϝ+       ��K	�U��A�"*

logging/current_cost�Y�;sU�X+       ��K	����A�"*

logging/current_cost�g�;���+       ��K	���A�"*

logging/current_cost�k�;��6�+       ��K	����A�"*

logging/current_cost�`�;��U+       ��K	�-��A�"*

logging/current_costb�;�Z?F+       ��K	9c��A�"*

logging/current_costgd�;WϽ�+       ��K	Ė��A�"*

logging/current_cost�a�;CX/+       ��K	P���A�"*

logging/current_cost[`�;'g�f+       ��K	s���A�"*

logging/current_cost�`�;��.+       ��K	�>��A�"*

logging/current_cost�_�;F��H+       ��K	�v��A�"*

logging/current_cost�]�;�ѐ�+       ��K	����A�"*

logging/current_cost�\�;��s+       ��K	����A�"*

logging/current_costYl�;��T�+       ��K	S��A�"*

logging/current_cost|o�;�C��+       ��K	�R��A�"*

logging/current_cost_�;I��+       ��K	J���A�"*

logging/current_cost�d�;Ǽs�+       ��K	����A�"*

logging/current_cost+S�;_��3+       ��K	P���A�"*

logging/current_costyn�;1���+       ��K	a���A�"*

logging/current_cost$^�;���d+       ��K	M���A�"*

logging/current_costKm�;g@�+       ��K	�����A�"*

logging/current_cost�c�;��+       ��K	4����A�"*

logging/current_cost�n�;�z��+       ��K	h����A�#*

logging/current_cost�W�;�
+       ��K	��A�#*

logging/current_costa�; �n+       ��K	E;��A�#*

logging/current_costyT�;j0�Y+       ��K	�p��A�#*

logging/current_costm�;2mo�+       ��K	����A�#*

logging/current_cost�h�;����+       ��K	u���A�#*

logging/current_cost`�;�b+       ��K	x	��A�#*

logging/current_costNc�;E�ɗ+       ��K	�6��A�#*

logging/current_costrd�;Պ�	+       ��K	 h��A�#*

logging/current_cost�Z�;z��+       ��K	R���A�#*

logging/current_costnc�;/�{+       ��K	S���A�#*

logging/current_cost�p�;��� +       ��K	����A�#*

logging/current_cost�c�;�v�+       ��K	+��A�#*

logging/current_cost�p�;½)�+       ��K	�Z��A�#*

logging/current_cost�R�;!ܟ+       ��K	ŋ��A�#*

logging/current_cost9d�;6o�+       ��K	G���A�#*

logging/current_costc�;���+       ��K	|���A�#*

logging/current_cost�Z�;�g˯+       ��K	q+��A�#*

logging/current_costdh�;y���+       ��K	�\��A�#*

logging/current_cost�h�;�)`+       ��K	6���A�#*

logging/current_costuj�;�'��+       ��K	����A�#*

logging/current_cost<p�;���+       ��K	x���A�#*

logging/current_cost�T�;k��+       ��K	�)��A�#*

logging/current_cost�c�;kɱ�+       ��K	�]��A�#*

logging/current_costNv�;XB�+       ��K	W���A�#*

logging/current_costGG�;7mq�+       ��K	����A�#*

logging/current_cost�q�;�/��+       ��K	���A�$*

logging/current_cost�F�;�J?O+       ��K	�I��A�$*

logging/current_costUt�;�3+       ��K	ŀ��A�$*

logging/current_costLl�;�?��+       ��K	G���A�$*

logging/current_cost@S�;��Ų+       ��K	����A�$*

logging/current_costy�;O>�+       ��K	!���A�$*

logging/current_cost�V�;�%&+       ��K	�V���A�$*

logging/current_cost^�;����+       ��K	S����A�$*

logging/current_cost�f�;o;r+       ��K	�����A�$*

logging/current_cost�c�;d��+       ��K	�����A�$*

logging/current_cost�`�;T
3�+       ��K	"*���A�$*

logging/current_costc�;֔m�+       ��K	RZ���A�$*

logging/current_cost�[�;�+       ��K	x����A�$*

logging/current_cost�n�;I��+       ��K	�����A�$*

logging/current_costpX�;�E��+       ��K	@����A�$*

logging/current_cost@R�;�1�+       ��K	����A�$*

logging/current_cost�m�;PM.	+       ��K	�C���A�$*

logging/current_cost�e�;e�&+       ��K	t���A�$*

logging/current_cost�h�;�y��+       ��K	����A�$*

logging/current_cost{Q�;�MHi+       ��K	�����A�$*

logging/current_costIl�;(�u�+       ��K	�����A�$*

logging/current_coste�;%S��+       ��K	.���A�$*

logging/current_cost�T�;s�\z+       ��K	\^���A�$*

logging/current_cost�m�;����+       ��K	؍���A�$*

logging/current_cost�U�;��+       ��K	����A�$*

logging/current_cost�t�;�T�+       ��K	����A�$*

logging/current_costui�;!J�+       ��K	}���A�%*

logging/current_cost�b�;�,3�+       ��K	�D���A�%*

logging/current_cost�W�;�2�+       ��K	�v���A�%*

logging/current_cost,c�;A���+       ��K	�����A�%*

logging/current_cost_�;��F+       ��K	J����A�%*

logging/current_cost�Y�;��'e+       ��K	����A�%*

logging/current_cost�j�;Y'��+       ��K	�N���A�%*

logging/current_cost�j�;���+       ��K	%����A�%*

logging/current_costrV�;����+       ��K	�����A�%*

logging/current_costQ�;|�a+       ��K	�����A�%*

logging/current_cost�e�;M؋�+       ��K	Z���A�%*

logging/current_cost.Z�;mJbn+       ��K	�S���A�%*

logging/current_cost{g�;t���+       ��K	i����A�%*

logging/current_cost�h�;%b+       ��K	A����A�%*

logging/current_cost�i�;/�j�+       ��K	x ���A�%*

logging/current_cost�\�;�͛@+       ��K	n2���A�%*

logging/current_cost�r�;��h�+       ��K	�h���A�%*

logging/current_cost�[�;d�+       ��K	9@���A�%*

logging/current_costM�;��Sv+       ��K	̃���A�%*

logging/current_cost�y�;h��;+       ��K	�����A�%*

logging/current_costNU�;ѓ�o+       ��K	U���A�%*

logging/current_costL]�;����+       ��K	zv���A�%*

logging/current_cost�U�;� x+       ��K	�����A�%*

logging/current_costX�;[Y֙+       ��K	�����A�%*

logging/current_cost�m�;��B_+       ��K	�2���A�%*

logging/current_cost�^�;q��4+       ��K	�n���A�&*

logging/current_cost�_�;:���+       ��K	�����A�&*

logging/current_cost�a�;��v�+       ��K	�����A�&*

logging/current_costTb�;���[+       ��K		���A�&*

logging/current_cost�U�;�>"+       ��K	�G���A�&*

logging/current_costl�;+�̲+       ��K	�{���A�&*

logging/current_costq�;-�+       ��K	�����A�&*

logging/current_costTI�;��1�+       ��K	F����A�&*

logging/current_costr�;�Ȗ_+       ��K	- ��A�&*

logging/current_cost�O�;�1r+       ��K	D ��A�&*

logging/current_cost�N�;��T+       ��K	r ��A�&*

logging/current_cost�q�;��D+       ��K	�� ��A�&*

logging/current_costuH�;J&�;+       ��K	�� ��A�&*

logging/current_cost���;oR�+       ��K	���A�&*

logging/current_cost9<�;�ʭ+       ��K	�3��A�&*

logging/current_costr]�;

Ԍ+       ��K	�a��A�&*

logging/current_costWq�;��8+       ��K	%���A�&*

logging/current_cost�a�;7ۄ�+       ��K	>���A�&*

logging/current_cost�i�;2��+       ��K	����A�&*

logging/current_cost"W�;K��_+       ��K	���A�&*

logging/current_cost�Y�;��Q�+       ��K	MM��A�&*

logging/current_cost9l�;���i+       ��K	<|��A�&*

logging/current_costg�;�Dcx+       ��K	����A�&*

logging/current_costUY�;d��}+       ��K	0���A�&*

logging/current_cost�c�;�*�+       ��K	���A�&*

logging/current_costJ�;���|+       ��K	2D��A�&*

logging/current_cost+w�;�J˭+       ��K	r��A�'*

logging/current_cost9Y�;;=��+       ��K	����A�'*

logging/current_costV�;��+       ��K	����A�'*

logging/current_cost b�;V��+       ��K	`���A�'*

logging/current_costO�;^�o+       ��K	�1��A�'*

logging/current_costrj�;���+       ��K	�b��A�'*

logging/current_cost�_�;R�?+       ��K	����A�'*

logging/current_cost�h�;r�I�+       ��K	Q���A�'*

logging/current_cost�W�;�p,+       ��K	.���A�'*

logging/current_cost�l�;6��+       ��K	K��A�'*

logging/current_cost�[�;=�k�+       ��K	�J��A�'*

logging/current_costL�;GN�M+       ��K	�z��A�'*

logging/current_costRY�;h�lW+       ��K	����A�'*

logging/current_costeU�;c^��+       ��K	J���A�'*

logging/current_cost�v�;�LV+       ��K	���A�'*

logging/current_costLD�;D?jg+       ��K	�9��A�'*

logging/current_cost�g�;�lG�+       ��K	�k��A�'*

logging/current_cost<[�;��x�+       ��K	D���A�'*

logging/current_cost�X�;
u�`+       ��K	M���A�'*

logging/current_cost�f�;���+       ��K	����A�'*

logging/current_cost�E�;��o�+       ��K	'��A�'*

logging/current_cost�m�;�L�+       ��K	MV��A�'*

logging/current_cost�Y�;dmp+       ��K	f���A�'*

logging/current_costR\�;r:+       ��K	���A�'*

logging/current_cost�P�;�@��+       ��K	k���A�'*

logging/current_costnr�;�6)+       ��K	(��A�(*

logging/current_cost R�;;`�+       ��K	RK��A�(*

logging/current_cost�Q�;[��+       ��K	{|��A�(*

logging/current_costk`�;Ul��+       ��K	���A�(*

logging/current_cost�`�;�XQ[+       ��K	����A�(*

logging/current_cost+U�;�o�+       ��K	/	��A�(*

logging/current_cost�O�;D!�,+       ��K	>f	��A�(*

logging/current_cost�a�;�#+       ��K	̓	��A�(*

logging/current_cost�d�;�w+�+       ��K	�	��A�(*

logging/current_cost2J�;,a�+       ��K	��	��A�(*

logging/current_cost�Y�;YoL&+       ��K	�+
��A�(*

logging/current_cost���;�E�i+       ��K	�`
��A�(*

logging/current_cost P�;��[�+       ��K	Ï
��A�(*

logging/current_cost�d�;�#�+       ��K	�
��A�(*

logging/current_cost�v�;���+       ��K	}�
��A�(*

logging/current_cost=�;���D+       ��K	 ��A�(*

logging/current_cost�n�;�q�#+       ��K	�K��A�(*

logging/current_cost�i�;;V�+       ��K	Ry��A�(*

logging/current_cost�H�;����+       ��K	C���A�(*

logging/current_cost�t�;��1C+       ��K	���A�(*

logging/current_costbT�;�	5�+       ��K	[��A�(*

logging/current_cost�c�;���O+       ��K	l8��A�(*

logging/current_cost~T�;���+       ��K	�h��A�(*

logging/current_cost`�;�ڭ�+       ��K	ŗ��A�(*

logging/current_cost�^�;5y��+       ��K	����A�(*

logging/current_costY4�;�f z+       ��K	����A�(*

logging/current_costf�;Huf�+       ��K	�!��A�)*

logging/current_costgb�; �v�+       ��K	�O��A�)*

logging/current_cost5f�;���+       ��K	}��A�)*

logging/current_cost�e�;i��g+       ��K	ߩ��A�)*

logging/current_cost~Z�;Bta+       ��K	c���A�)*

logging/current_cost�q�;<ث+       ��K	���A�)*

logging/current_costnN�;��H+       ��K	�7��A�)*

logging/current_cost26�;S�?�+       ��K	,g��A�)*

logging/current_cost�h�;cy(+       ��K	���A�)*

logging/current_cost�e�;L�8y+       ��K	o���A�)*

logging/current_cost9_�;$��O+       ��K	h���A�)*

logging/current_cost�F�;t��6+       ��K	� ��A�)*

logging/current_costn�;[K��+       ��K	L��A�)*

logging/current_cost�W�;q?�+       ��K	4���A�)*

logging/current_cost�O�;���+       ��K	����A�)*

logging/current_cost��; p�]+       ��K	���A�)*

logging/current_cost�3�;ߡC�+       ��K	�K��A�)*

logging/current_costLL�;���g+       ��K	g���A�)*

logging/current_costˀ�;~�<w+       ��K	Ƶ��A�)*

logging/current_costG`�;�k+       ��K	����A�)*

logging/current_cost�C�;��3W+       ��K	4&��A�)*

logging/current_cost)n�;�J�+       ��K	B]��A�)*

logging/current_cost�g�;�j��+       ��K	!���A�)*

logging/current_cost�P�;���+       ��K	����A�)*

logging/current_cost9T�;}��P+       ��K	����A�)*

logging/current_cost�`�;�$��+       ��K	r1��A�)*

logging/current_cost�[�;#pP�+       ��K	ma��A�**

logging/current_cost`L�;Sf+       ��K	���A�**

logging/current_cost�o�;M��T+       ��K	����A�**

logging/current_cost�?�;�x�+       ��K	����A�**

logging/current_cost�P�;��$+       ��K	�#��A�**

logging/current_cost||�;�J��+       ��K	�T��A�**

logging/current_costD8�;R2s+       ��K	����A�**

logging/current_cost p�;���+       ��K	3���A�**

logging/current_cost�R�;S��+       ��K	+���A�**

logging/current_costU�;ݘ�+       ��K	f-��A�**

logging/current_costNf�;]�C�+       ��K	%k��A�**

logging/current_cost�E�;��+       ��K	G���A�**

logging/current_cost�{�;�ԓ�+       ��K	,���A�**

logging/current_coste`�;x��+       ��K	�'��A�**

logging/current_cost�@�;��OO+       ��K	#_��A�**

logging/current_cost�R�;*�2+       ��K	E���A�**

logging/current_costdU�;k9+�+       ��K	����A�**

logging/current_cost�^�;>+       ��K	��A�**

logging/current_costrO�;��4�+       ��K	�]��A�**

logging/current_cost7]�;6��+       ��K	����A�**

logging/current_cost�c�;�"�!+       ��K	���A�**

logging/current_cost�I�;���+       ��K	2`��A�**

logging/current_cost�]�;���1+       ��K	���A�**

logging/current_cost�s�;�N�?+       ��K	B���A�**

logging/current_cost�M�;;-��+       ��K	��A�**

logging/current_cost f�;��h�+       ��K	mB��A�+*

logging/current_coste�;��<+       ��K	k���A�+*

logging/current_cost�<�;�Y�i+       ��K	����A�+*

logging/current_cost$j�;�x$2+       ��K	����A�+*

logging/current_cost�l�;9��+       ��K	J9��A�+*

logging/current_costNQ�;s��D+       ��K	�r��A�+*

logging/current_cost�W�;���y+       ��K	����A�+*

logging/current_cost�R�;;"��+       ��K	���A�+*

logging/current_costh�;e�m+       ��K	�.��A�+*

logging/current_cost�l�;�:�5+       ��K	vb��A�+*

logging/current_cost c�;��
+       ��K	ś��A�+*

logging/current_cost Z�;���+       ��K	����A�+*

logging/current_cost�G�;�JU+       ��K	���A�+*

logging/current_cost<K�;7��+       ��K	oR��A�+*

logging/current_cost�V�;��i+       ��K	����A�+*

logging/current_cost�q�;0m�+       ��K	_���A�+*

logging/current_cost\N�;��2+       ��K	��A�+*

logging/current_cost�9�;� �+       ��K	c��A�+*

logging/current_cost�;����+       ��K	h���A�+*

logging/current_cost�T�;���{+       ��K	"��A�+*

logging/current_cost�:�;�=�F+       ��K	OE��A�+*

logging/current_cost�b�;��+       ��K	���A�+*

logging/current_cost�U�;� N&+       ��K	<���A�+*

logging/current_cost V�;�<w�+       ��K	����A�+*

logging/current_coste�;G��+       ��K	�.��A�+*

logging/current_costu^�;P���+       ��K	.`��A�+*

logging/current_cost�E�;_x"+       ��K	����A�,*

logging/current_cost�H�;�\��+       ��K	 ���A�,*

logging/current_costb�;�m�k+       ��K	;��A�,*

logging/current_costS�;�E�+       ��K	�1��A�,*

logging/current_costF�;+Ly�+       ��K	j��A�,*

logging/current_cost�O�;�+       ��K	޽��A�,*

logging/current_cost<h�;}(�+       ��K	����A�,*

logging/current_cost�o�;�.��+       ��K	�! ��A�,*

logging/current_cost;W�;ݷ�+       ��K	�T ��A�,*

logging/current_costO�;о�|+       ��K	c� ��A�,*

logging/current_costYW�;@���+       ��K	�� ��A�,*

logging/current_cost\�;,�p�+       ��K	{� ��A�,*

logging/current_coste]�;�n�+       ��K	�/!��A�,*

logging/current_cost�b�;���o+       ��K	C`!��A�,*

logging/current_cost�j�;��9q+       ��K	,�!��A�,*

logging/current_costDC�;-;܌+       ��K	~�!��A�,*

logging/current_costi:�;�wG�+       ��K	��!��A�,*

logging/current_cost�T�;ړ2�+       ��K	�*"��A�,*

logging/current_cost�D�;���k+       ��K	^"��A�,*

logging/current_cost�:�;ǭ��+       ��K	-�"��A�,*

logging/current_costNs�;�D�+       ��K	��"��A�,*

logging/current_cost�{�;�#�V+       ��K	�#��A�,*

logging/current_costW�;�i9`+       ��K	@#��A�,*

logging/current_cost�Z�;��E+       ��K	Zq#��A�,*

logging/current_costw\�;Fѻ�+       ��K	�#��A�,*

logging/current_costkI�;��k+       ��K	�#��A�-*

logging/current_cost'>�;�	�C+       ��K	�$��A�-*

logging/current_cost�Q�;�%\�+       ��K	M$��A�-*

logging/current_costm�;��6+       ��K	�~$��A�-*

logging/current_cost�e�;LK\�+       ��K	��$��A�-*

logging/current_cost�1�;����+       ��K	��$��A�-*

logging/current_cost�9�;���+       ��K	 
%��A�-*

logging/current_costUY�;�}��+       ��K	B6%��A�-*

logging/current_cost�Z�;�d��+       ��K	�d%��A�-*

logging/current_cost�J�;��`�+       ��K	��%��A�-*

logging/current_cost�a�;- � +       ��K	<�%��A�-*

logging/current_cost�j�;%X|/+       ��K	��%��A�-*

logging/current_costi�;�G]w+       ��K	�&��A�-*

logging/current_cost�I�;GI�O+       ��K	�L&��A�-*

logging/current_costH�;�I@+       ��K	�y&��A�-*

logging/current_cost�h�;��k�+       ��K	��&��A�-*

logging/current_cost;k�;Îa�+       ��K	�&��A�-*

logging/current_costK�;X�~�+       ��K	�	'��A�-*

logging/current_costg?�;�G�+       ��K	�8'��A�-*

logging/current_cost�Y�;,�ӱ+       ��K	|e'��A�-*

logging/current_cost9z�;�()P+       ��K	��'��A�-*

logging/current_cost]�;�� 7+       ��K	��'��A�-*

logging/current_cost�K�;��g2+       ��K	��'��A�-*

logging/current_cost�q�;R\{�+       ��K	(��A�-*

logging/current_cost\�;��JN+       ��K	M(��A�-*

logging/current_cost�&�;�iJ<+       ��K	�y(��A�-*

logging/current_cost�>�;?���+       ��K	��(��A�.*

logging/current_cost,O�;5��/+       ��K	��(��A�.*

logging/current_costIG�;�P\+       ��K	_)��A�.*

logging/current_cost%j�;l}�+       ��K	�0)��A�.*

logging/current_cost�y�;/���+       ��K	�_)��A�.*

logging/current_cost�^�;�p'+       ��K	�)��A�.*

logging/current_cost�S�;ﮅ�+       ��K	��)��A�.*

logging/current_cost�[�;�c1\+       ��K	��)��A�.*

logging/current_costO�;D�)n+       ��K	�*��A�.*

logging/current_cost^5�;c;�+       ��K	�B*��A�.*

logging/current_cost�D�;��9�+       ��K	�p*��A�.*

logging/current_cost�K�;��lZ+       ��K	��*��A�.*

logging/current_cost;]�;{���+       ��K	��*��A�.*

logging/current_cost�j�;w�\�+       ��K	�*��A�.*

logging/current_costDJ�;��F+       ��K	�4+��A�.*

logging/current_cost�X�;i��+       ��K	yd+��A�.*

logging/current_cost�V�;���+       ��K	}�+��A�.*

logging/current_costb�;�E�+       ��K	��+��A�.*

logging/current_cost0]�;����+       ��K	V�+��A�.*

logging/current_costb.�;(�Z+       ��K	�,��A�.*

logging/current_cost0=�;A�H+       ��K	;K,��A�.*

logging/current_cost<y�;�C+       ��K	3z,��A�.*

logging/current_cost�i�;G4b�+       ��K	��,��A�.*

logging/current_cost77�;d���+       ��K	r�,��A�.*

logging/current_costDR�;\'+       ��K	�-��A�.*

logging/current_cost�k�;�,+       ��K	:-��A�.*

logging/current_costWc�;����+       ��K	ni-��A�/*

logging/current_cost�k�;��+       ��K	B�-��A�/*

logging/current_costiI�;ƛQ\+       ��K	��-��A�/*

logging/current_cost~-�;PS�f+       ��K	��-��A�/*

logging/current_costH�;�3� +       ��K	�#.��A�/*

logging/current_costI^�;��y�+       ��K	OS.��A�/*

logging/current_cost�e�;΅�L+       ��K	�.��A�/*

logging/current_cost�C�;00��+       ��K	��.��A�/*

logging/current_cost�3�;��m�+       ��K	��.��A�/*

logging/current_cost�D�;փ��+       ��K	/��A�/*

logging/current_cost�^�;ki�+       ��K	�=/��A�/*

logging/current_cost�V�;����+       ��K	�l/��A�/*

logging/current_costuH�;l���+       ��K	'�/��A�/*

logging/current_cost�V�;R�39+       ��K	x�/��A�/*

logging/current_cost�d�;}(�@+       ��K	��/��A�/*

logging/current_cost�@�;:�S+       ��K	'0��A�/*

logging/current_costU8�;�>)+       ��K	�Z0��A�/*

logging/current_cost�R�;l�6�+       ��K	��0��A�/*

logging/current_costdw�;��o�+       ��K	 �0��A�/*

logging/current_cost�c�;��O�+       ��K	��0��A�/*

logging/current_costG�;<�*z+       ��K	[1��A�/*

logging/current_costb5�;��81+       ��K	�J1��A�/*

logging/current_cost�I�;�5?�+       ��K	�x1��A�/*

logging/current_costK_�;<��'+       ��K	.�1��A�/*

logging/current_costRo�;��\�+       ��K	��1��A�/*

logging/current_cost`f�;�BS+       ��K	�2��A�0*

logging/current_cost$N�;b�++       ��K	y32��A�0*

logging/current_cost)H�;��WQ+       ��K	�a2��A�0*

logging/current_costh�;�^+       ��K	��2��A�0*

logging/current_costg�;~���+       ��K	��2��A�0*

logging/current_cost�_�;V� 4+       ��K	��2��A�0*

logging/current_costi�;Luf+       ��K	�3��A�0*

logging/current_costUN�;tA�+       ��K	pH3��A�0*

logging/current_cost�B�;��α+       ��K	�u3��A�0*

logging/current_costN�;^ϩ<+       ��K	0�3��A�0*

logging/current_cost�U�;o��E+       ��K	0�3��A�0*

logging/current_cost5_�;�p�+       ��K	�4��A�0*

logging/current_costUS�;�[+       ��K	�54��A�0*

logging/current_costi(�;����+       ��K	Wg4��A�0*

logging/current_cost2�;�78�+       ��K	n�4��A�0*

logging/current_cost�r�;��-Q+       ��K	Y�4��A�0*

logging/current_costp|�;	��+       ��K	�4��A�0*

logging/current_cost[�;��z+       ��K	�5��A�0*

logging/current_costK?�;���+       ��K	vS5��A�0*

logging/current_costDB�;��+       ��K	!�5��A�0*

logging/current_cost�a�;�U��+       ��K	[�5��A�0*

logging/current_costyd�;�!��+       ��K	��5��A�0*

logging/current_cost�k�;⠨�+       ��K	�6��A�0*

logging/current_costKO�;^
��+       ��K	ZW6��A�0*

logging/current_costB+�;�7�+       ��K	r�6��A�0*

logging/current_cost<Y�;����+       ��K	T�6��A�0*

logging/current_costiZ�;���+       ��K	+7��A�1*

logging/current_coste`�;ld:Q+       ��K	�d7��A�1*

logging/current_cost�>�;5�y�+       ��K	��7��A�1*

logging/current_costA�;Zg�+       ��K	|�7��A�1*

logging/current_cost{[�;�5n+       ��K	��7��A�1*

logging/current_cost5g�;#=�+       ��K	18��A�1*

logging/current_costk`�;+���+       ��K	Ry8��A�1*

logging/current_cost�I�;2~�1+       ��K	[�8��A�1*

logging/current_costlF�;����+       ��K	@�8��A�1*

logging/current_cost�W�;�)�B+       ��K	�)9��A�1*

logging/current_cost]�;�^k�+       ��K	�\9��A�1*

logging/current_cost�B�;� ��+       ��K	�9��A�1*

logging/current_cost�+�;f-@F+       ��K	f�9��A�1*

logging/current_cost@=�;dO�+       ��K	�:��A�1*

logging/current_cost�a�;���+       ��K	_6:��A�1*

logging/current_cost�[�;�r�M+       ��K	�d:��A�1*

logging/current_costUF�;I���+       ��K	��:��A�1*

logging/current_cost�3�;��O�+       ��K	.�:��A�1*

logging/current_cost�L�;�A�.+       ��K	��:��A�1*

logging/current_cost�T�;Cs�e+       ��K	�#;��A�1*

logging/current_cost�~�;��:�+       ��K	U;��A�1*

logging/current_cost�B�;El��+       ��K	*�;��A�1*

logging/current_cost�3�;c�)�+       ��K	�<��A�1*

logging/current_cost�?�;&{:+       ��K	i:<��A�1*

logging/current_costTk�;�W��+       ��K	�l<��A�1*

logging/current_cost�p�;wa��+       ��K	_�<��A�2*

logging/current_cost�o�;����+       ��K	��<��A�2*

logging/current_cost	Y�;DZ�*+       ��K	=��A�2*

logging/current_cost5C�;�7)�+       ��K	�9=��A�2*

logging/current_cost"K�;��i+       ��K	�h=��A�2*

logging/current_cost�U�;[�+       ��K	Ė=��A�2*

logging/current_costi_�;"��+       ��K	��=��A�2*

logging/current_costiu�;��͞+       ��K	+�=��A�2*

logging/current_cost�V�;mg.+       ��K	�0>��A�2*

logging/current_costgD�;�� �+       ��K	xc>��A�2*

logging/current_cost~B�;�Od8+       ��K	/�>��A�2*

logging/current_cost�N�; <��+       ��K	3�>��A�2*

logging/current_cost	g�;z&��+       ��K	�?��A�2*

logging/current_cost~m�;����+       ��K	/H?��A�2*

logging/current_cost�Z�;�F
o+       ��K	�y?��A�2*

logging/current_cost L�;��N+       ��K	��?��A�2*

logging/current_costr]�;�*J�+       ��K	H�?��A�2*

logging/current_cost�b�;#)��+       ��K	@@��A�2*

logging/current_cost�[�;5��+       ��K	5@��A�2*

logging/current_costQ�;��	+       ��K	+a@��A�2*

logging/current_costrF�;����+       ��K	W�@��A�2*

logging/current_costw3�;�؅+       ��K	L�@��A�2*

logging/current_cost{4�;��+       ��K	��@��A�2*

logging/current_cost�?�;��b0+       ��K	�A��A�2*

logging/current_cost�K�;�{�+       ��K	�EA��A�2*

logging/current_cost�Y�;%,�+       ��K	�tA��A�2*

logging/current_cost�h�;VT�++       ��K	1�A��A�3*

logging/current_costGr�;��V+       ��K	!�A��A�3*

logging/current_cost%g�;)]�+       ��K	��A��A�3*

logging/current_costm�;�0�*+       ��K	�*B��A�3*

logging/current_cost�P�;X��
+       ��K	�XB��A�3*

logging/current_cost23�;`��+       ��K	J�B��A�3*

logging/current_cost2(�;�.2"+       ��K	��B��A�3*

logging/current_costy2�;=��s+       ��K	q�B��A�3*

logging/current_cost�V�;�Ԫ�+       ��K	�C��A�3*

logging/current_costb\�;���s+       ��K	)LC��A�3*

logging/current_cost�C�;^��k+       ��K	�{C��A�3*

logging/current_cost3�;�7H+       ��K	z�C��A�3*

logging/current_cost�N�;�W�+       ��K	P�C��A�3*

logging/current_cost<b�;b�7�+       ��K	FD��A�3*

logging/current_cost�a�;�]:�+       ��K	�0D��A�3*

logging/current_costPG�;�1�2+       ��K	�^D��A�3*

logging/current_cost+�;a@39+       ��K	��D��A�3*

logging/current_costY6�;&��+       ��K	1�D��A�3*

logging/current_costEB�;2Rj�+       ��K	��D��A�3*

logging/current_costDZ�;s��+       ��K	�E��A�3*

logging/current_cost�Y�;w��O+       ��K	3LE��A�3*

logging/current_cost�C�;���+       ��K	E��A�3*

logging/current_cost4�;��*7+       ��K	��E��A�3*

logging/current_cost�+�;W��+       ��K	�E��A�3*

logging/current_costI8�;��T�+       ��K	;F��A�3*

logging/current_cost|Y�;��+       ��K	F;F��A�3*

logging/current_cost�k�;�O��+       ��K	�hF��A�4*

logging/current_cost }�;��J�+       ��K	B�F��A�4*

logging/current_cost�v�;��(�+       ��K	��F��A�4*

logging/current_cost�y�;���+       ��K	��F��A�4*

logging/current_cost�N�;���+       ��K	r!G��A�4*

logging/current_cost�'�;���E+       ��K	�OG��A�4*

logging/current_cost�!�;UJ�\+       ��K	�}G��A�4*

logging/current_cost1�;u�+       ��K	ۭG��A�4*

logging/current_cost4Y�;T�+       ��K	1�G��A�4*

logging/current_costW\�;�ۥ+       ��K	�H��A�4*

logging/current_costuG�;�ۓ+       ��K	m8H��A�4*

logging/current_costa�;�I/�+       ��K	�kH��A�4*

logging/current_cost,6�;z��M+       ��K	�H��A�4*

logging/current_cost�G�;��+       ��K	��H��A�4*

logging/current_cost�1�;��|+       ��K	I��A�4*

logging/current_cost�R�;�f��+       ��K	-PI��A�4*

logging/current_cost\D�;�Q��+       ��K	p�I��A�4*

logging/current_cost�M�;���+       ��K	�I��A�4*

logging/current_costN�;�؟�+       ��K	�J��A�4*

logging/current_costy8�;���+       ��K	�LJ��A�4*

logging/current_cost�L�;V�1+       ��K	ÐJ��A�4*

logging/current_costtZ�;�6i+       ��K	��J��A�4*

logging/current_costr`�;Y�V�+       ��K	�K��A�4*

logging/current_cost��;t���+       ��K	�AK��A�4*

logging/current_cost�t�;%�+       ��K	OxK��A�4*

logging/current_cost L�;���e+       ��K		�K��A�5*

logging/current_cost�[�;W��G+       ��K	��K��A�5*

logging/current_cost�6�;��+       ��K	;L��A�5*

logging/current_cost�M�;WC��+       ��K	�SL��A�5*

logging/current_cost�3�;�a��+       ��K	�L��A�5*

logging/current_cost�K�;��+       ��K	��L��A�5*

logging/current_cost�R�;�j��+       ��K	�L��A�5*

logging/current_cost$N�;��] +       ��K	!!M��A�5*

logging/current_cost"p�;�@S+       ��K	�NM��A�5*

logging/current_cost+X�;U
9�+       ��K	�~M��A�5*

logging/current_cost�X�;<���+       ��K	�M��A�5*

logging/current_cost+Z�;u���+       ��K	^�M��A�5*

logging/current_cost�R�;����+       ��K	N��A�5*

logging/current_cost�D�;p=�D+       ��K	�=N��A�5*

logging/current_cost@,�;�±v+       ��K	nN��A�5*

logging/current_costn6�;�j+       ��K	ϝN��A�5*

logging/current_cost�7�;	�7+       ��K	��N��A�5*

logging/current_costKR�;!栻+       ��K	{�N��A�5*

logging/current_cost|e�;�T�e+       ��K	�(O��A�5*

logging/current_cost ;�;	`n+       ��K	�UO��A�5*

logging/current_cost�W�;'�4�+       ��K	Q�O��A�5*

logging/current_cost�_�;��$�+       ��K	b�O��A�5*

logging/current_cost�]�;u-*�+       ��K	<�O��A�5*

logging/current_cost�Q�;+�g�+       ��K	PP��A�5*

logging/current_cost�E�;W�+       ��K	�LP��A�5*

logging/current_cost�6�;�c$+       ��K	�zP��A�5*

logging/current_cost5>�;��+       ��K	��P��A�6*

logging/current_costrF�;����+       ��K	j�P��A�6*

logging/current_cost�D�;���H+       ��K	Q��A�6*

logging/current_cost�L�;S�B+       ��K	(?Q��A�6*

logging/current_costDd�;��� +       ��K	�lQ��A�6*

logging/current_costrT�;ǭ^+       ��K	ʚQ��A�6*

logging/current_cost<K�;2��[+       ��K	b�Q��A�6*

logging/current_costDI�;H�9+       ��K	��Q��A�6*

logging/current_cost�B�;F��y+       ��K	Q(R��A�6*

logging/current_cost6�;����+       ��K	cZR��A�6*

logging/current_costgI�;��V7+       ��K	%�R��A�6*

logging/current_costt3�;���+       ��K	��R��A�6*

logging/current_costeT�;�W��+       ��K	��R��A�6*

logging/current_costUG�;�tMR+       ��K	&'S��A�6*

logging/current_costKL�;+%��+       ��K	�US��A�6*

logging/current_cost<J�;@�0P+       ��K	L�S��A�6*

logging/current_costL�;���[+       ��K	��S��A�6*

logging/current_cost�f�;�l1�+       ��K	c�S��A�6*

logging/current_cost�O�;�y�+       ��K	"T��A�6*

logging/current_cost�S�;L��u+       ��K	tVT��A�6*

logging/current_cost�6�;X�+       ��K	Y�T��A�6*

logging/current_cost[D�;���+       ��K	=�T��A�6*

logging/current_cost�F�;�T`�+       ��K	U��A�6*

logging/current_cost	D�;]a�+       ��K	4<U��A�6*

logging/current_cost7a�;����+       ��K	�rU��A�6*

logging/current_cost�d�;���+       ��K	�U��A�7*

logging/current_cost�^�;��ն+       ��K	4�U��A�7*

logging/current_cost>V�;�<+       ��K	eV��A�7*

logging/current_cost@a�;��+       ��K	�RV��A�7*

logging/current_cost�R�;˖+       ��K	Z�V��A�7*

logging/current_costV�;I��^+       ��K	ŲV��A�7*

logging/current_costx�;���+       ��K	��V��A�7*

logging/current_cost�}�;��+       ��K	�W��A�7*

logging/current_cost$w�;���v+       ��K	�MW��A�7*

logging/current_cost�f�;a:h�+       ��K	"}W��A�7*

logging/current_cost�w�;��E+       ��K	Q�W��A�7*

logging/current_costGh�;g_+]+       ��K	0�W��A�7*

logging/current_cost�Z�;?;+       ��K	OX��A�7*

logging/current_cost�W�;���5+       ��K	�BX��A�7*

logging/current_cost+\�;���Q+       ��K	�sX��A�7*

logging/current_cost�n�;7�� +       ��K	+�X��A�7*

logging/current_cost�@�;��9+       ��K	��X��A�7*

logging/current_costyW�;4��=+       ��K	�Y��A�7*

logging/current_cost�"�;�{&�+       ��K	�6Y��A�7*

logging/current_cost�X�;�Q�+       ��K	LjY��A�7*

logging/current_cost51�;�b��+       ��K	�Y��A�7*

logging/current_costbR�;�s�+       ��K	��Y��A�7*

logging/current_cost�S�;(ݴ�+       ��K	��Y��A�7*

logging/current_costeE�;����+       ��K	6-Z��A�7*

logging/current_cost�,�;�l+�+       ��K	�^Z��A�7*

logging/current_costI�;o1��+       ��K	ÎZ��A�7*

logging/current_cost�R�;�H�+       ��K	��Z��A�8*

logging/current_cost�L�;#\�+       ��K	W�Z��A�8*

logging/current_cost4V�;����+       ��K	#[��A�8*

logging/current_cost�e�;�$��+       ��K	b[��A�8*

logging/current_cost�k�;x!5�+       ��K	ʹ[��A�8*

logging/current_costG�;����+       ��K	��[��A�8*

logging/current_costQ�;h���+       ��K	�"\��A�8*

logging/current_cost�D�;	</+       ��K	�X\��A�8*

logging/current_cost�Q�;�ef#+       ��K	%�\��A�8*

logging/current_costY�;6�	�+       ��K	 �\��A�8*

logging/current_cost+Y�;�n9�+       ��K	k]��A�8*

logging/current_cost�S�;����+       ��K	}?]��A�8*

logging/current_costpK�;)SU+       ��K	$u]��A�8*

logging/current_cost�?�;v��+       ��K	p�]��A�8*

logging/current_cost��;���+       ��K	��]��A�8*

logging/current_cost�;�'�T+       ��K	]$^��A�8*

logging/current_cost�1�;릓+       ��K	�_^��A�8*

logging/current_cost�2�;�:�+       ��K	n�^��A�8*

logging/current_cost�H�;;��A+       ��K	��^��A�8*

logging/current_cost V�;[ͩ�+       ��K	�_��A�8*

logging/current_cost��;��p�+       ��K	�C_��A�8*

logging/current_cost�J�;��q+       ��K	�~_��A�8*

logging/current_cost|W�;�@"�+       ��K	��_��A�8*

logging/current_cost�]�;�_G�+       ��K	L�_��A�8*

logging/current_cost�E�;��+       ��K	�!`��A�8*

logging/current_cost5Z�;��(�+       ��K	
^`��A�8*

logging/current_costi�;�0�+       ��K	�`��A�9*

logging/current_cost�O�;����+       ��K	��`��A�9*

logging/current_cost�2�;b=+       ��K	
a��A�9*

logging/current_costg:�;>�*+       ��K	�@a��A�9*

logging/current_costI'�;���+       ��K	�ua��A�9*

logging/current_cost�C�;����+       ��K	3�a��A�9*

logging/current_cost�o�;��4,+       ��K	0�a��A�9*

logging/current_cost�n�;�T��+       ��K	ib��A�9*

logging/current_cost�Z�;ܡ�+       ��K	�Bb��A�9*

logging/current_costnW�;�c�+       ��K	�ob��A�9*

logging/current_cost 0�;� ��+       ��K	�b��A�9*

logging/current_cost*�;��+       ��K	��b��A�9*

logging/current_cost�D�;�P�M+       ��K	|c��A�9*

logging/current_cost�I�;˂[+       ��K	e1c��A�9*

logging/current_costU[�;f�u+       ��K	�`c��A�9*

logging/current_costV�;Qeۡ+       ��K	ɏc��A�9*

logging/current_cost�?�;E�m+       ��K	��c��A�9*

logging/current_costYk�;��D+       ��K	�c��A�9*

logging/current_cost 0�;X��i+       ��K	8d��A�9*

logging/current_cost�Z�;:v�@+       ��K	�Ld��A�9*

logging/current_cost�<�;H��#+       ��K	�{d��A�9*

logging/current_cost�b�;*N�q+       ��K	Ӧd��A�9*

logging/current_cost'[�;@�.M+       ��K	��d��A�9*

logging/current_costC�;v�6Z+       ��K		e��A�9*

logging/current_cost�D�;�D�E+       ��K	�:e��A�9*

logging/current_cost�$�;�Ƚ+       ��K	bje��A�:*

logging/current_costn&�;�?9�+       ��K	՘e��A�:*

logging/current_cost�J�;�k0T+       ��K	��e��A�:*

logging/current_cost�L�;�5�+       ��K	~�e��A�:*

logging/current_cost$h�;�F7+       ��K	�&f��A�:*

logging/current_cost�n�;8d�+       ��K	zWf��A�:*

logging/current_costp�;[���+       ��K	�f��A�:*

logging/current_cost�3�;�;+       ��K	T�f��A�:*

logging/current_costy1�;���+       ��K	��f��A�:*

logging/current_cost�C�;p\ݽ+       ��K	�g��A�:*

logging/current_cost�V�;%��!+       ��K	�Bg��A�:*

logging/current_cost�^�;�¾�+       ��K	drg��A�:*

logging/current_costI[�;{�+       ��K	2�g��A�:*

logging/current_cost�I�;��f+       ��K	/�g��A�:*

logging/current_cost$,�;���L+       ��K	�g��A�:*

logging/current_cost %�;J<�q+       ��K	�,h��A�:*

logging/current_cost�1�;��f+       ��K	�Zh��A�:*

logging/current_cost`Y�;��+       ��K	<�h��A�:*

logging/current_cost�[�;b
�+       ��K	��h��A�:*

logging/current_cost%D�;��+       ��K	��h��A�:*

logging/current_cost�0�;5	�+       ��K	Ci��A�:*

logging/current_cost�I�;�+       ��K	�>i��A�:*

logging/current_costH�;ù��+       ��K	�mi��A�:*

logging/current_cost�P�;Gd��+       ��K	��i��A�:*

logging/current_costlB�;UOI�+       ��K	��i��A�:*

logging/current_cost�=�;��:+       ��K	g�i��A�:*

logging/current_cost�4�;��Q+       ��K	�&j��A�;*

logging/current_cost+U�;��O�+       ��K	�Sj��A�;*

logging/current_cost�a�;n��+       ��K	��j��A�;*

logging/current_cost �;x<��+       ��K	t�j��A�;*

logging/current_cost�Q�;�W+       ��K	��j��A�;*

logging/current_cost�W�;ڝ�"+       ��K	�	k��A�;*

logging/current_costA�;��K+       ��K	f6k��A�;*

logging/current_costk4�;ض��+       ��K	�ck��A�;*

logging/current_cost�S�;A�-�+       ��K	+�k��A�;*

logging/current_cost�G�;`��H+       ��K	��k��A�;*

logging/current_costNW�;^@4S+       ��K	��k��A�;*

logging/current_cost \�;���+       ��K	�l��A�;*

logging/current_costJ�;8��h+       ��K	�Ll��A�;*

logging/current_cost\�;��S�+       ��K	P{l��A�;*

logging/current_cost�S�;P.+       ��K	H�l��A�;*

logging/current_cost3�;y�7+       ��K	2�l��A�;*

logging/current_cost`P�;Y�z+       ��K	#m��A�;*

logging/current_costiG�;r��=+       ��K	�9m��A�;*

logging/current_cost�7�;	o�D+       ��K	�gm��A�;*

logging/current_costf�;����+       ��K	l�m��A�;*

logging/current_cost+T�;��a+       ��K	��m��A�;*

logging/current_costL)�;� �+       ��K	1�m��A�;*

logging/current_cost�d�;�[q+       ��K	�"n��A�;*

logging/current_cost�"�;�� +       ��K	�Tn��A�;*

logging/current_costl�;�-�+       ��K	2�n��A�;*

logging/current_cost�4�;.�J�+       ��K	|�n��A�<*

logging/current_costnn�;�͹�+       ��K	
�n��A�<*

logging/current_cost%a�;�C$�+       ��K	Do��A�<*

logging/current_cost�/�;��W�+       ��K	<:o��A�<*

logging/current_costYj�;6�Ӱ+       ��K	#go��A�<*

logging/current_costuH�;O Vn+       ��K	ŗo��A�<*

logging/current_cost^S�;^&d+       ��K	��o��A�<*

logging/current_cost�W�;Q��T+       ��K	��o��A�<*

logging/current_cost�G�;�>u+       ��K	)'p��A�<*

logging/current_costu9�; �e+       ��K	�Wp��A�<*

logging/current_cost�7�;����+       ��K	��p��A�<*

logging/current_cost�a�;(�H�+       ��K	ָp��A�<*

logging/current_costn`�;}?�+       ��K	J�p��A�<*

logging/current_costLe�;��}D+       ��K	+q��A�<*

logging/current_costN�;��I�+       ��K	dJq��A�<*

logging/current_cost�4�;��+       ��K	xq��A�<*

logging/current_cost��;4��+       ��K	?�q��A�<*

logging/current_costRG�;Y��/+       ��K	]�q��A�<*

logging/current_cost)y�;�e�>+       ��K	�r��A�<*

logging/current_cost�s�;���w+       ��K	,=r��A�<*

logging/current_cost|e�;�b�~+       ��K	�ir��A�<*

logging/current_cost�<�;K�HO+       ��K	��r��A�<*

logging/current_cost��;P��+       ��K	��r��A�<*

logging/current_cost7�;uf�+       ��K	(�r��A�<*

logging/current_costS�; ��0+       ��K	)/s��A�<*

logging/current_costY\�;���+       ��K	�\s��A�<*

logging/current_costw{�;��ø+       ��K	��s��A�=*

logging/current_cost�\�;	�B9+       ��K	��s��A�=*

logging/current_costn?�;7]�+       ��K	��s��A�=*

logging/current_cost�H�;�ntv+       ��K	)t��A�=*

logging/current_costO�;��h+       ��K	�]t��A�=*

logging/current_cost�,�;��1�+       ��K	�t��A�=*

logging/current_cost�G�;N�ק+       ��K	+�t��A�=*

logging/current_cost.w�;E+       ��K	�t��A�=*

logging/current_costRb�;�n�+       ��K	`#u��A�=*

logging/current_cost�L�;��g+       ��K	�Ru��A�=*

logging/current_cost�*�;K^+       ��K	��u��A�=*

logging/current_cost�7�;7�H�+       ��K	��u��A�=*

logging/current_cost�l�;E��O+       ��K	w�u��A�=*

logging/current_cost�H�;wݐ�+       ��K	�v��A�=*

logging/current_costQ�;3+       ��K	�Bv��A�=*

logging/current_cost�U�;u{��+       ��K	�rv��A�=*

logging/current_costg;�;qi�+       ��K	��v��A�=*

logging/current_cost�H�;־@+       ��K	]�v��A�=*

logging/current_cost'*�;B+       ��K	��v��A�=*

logging/current_cost�p�;c�+       ��K	�+w��A�=*

logging/current_costN<�;h��+       ��K	[Yw��A�=*

logging/current_costR�;��v�+       ��K	��w��A�=*

logging/current_cost�L�;q6�+       ��K	n�w��A�=*

logging/current_costUq�;>eAJ+       ��K	��w��A�=*

logging/current_costWQ�;G̢+       ��K	1x��A�=*

logging/current_cost")�;rp +       ��K	�Bx��A�=*

logging/current_cost� �;9�5+       ��K	�px��A�>*

logging/current_cost�\�;����+       ��K	8�x��A�>*

logging/current_costn:�;3I�+       ��K	8�x��A�>*

logging/current_cost���;$�D+       ��K	�x��A�>*

logging/current_cost�%�;;C#+       ��K	V,y��A�>*

logging/current_costNO�;ݚ��+       ��K	f[y��A�>*

logging/current_costg1�;@/Y\+       ��K	��y��A�>*

logging/current_cost�=�;��G�+       ��K	l�y��A�>*

logging/current_cost�W�;H���+       ��K	��y��A�>*

logging/current_cost�0�;�vf�+       ��K	�z��A�>*

logging/current_cost�K�;(���+       ��K	AQz��A�>*

logging/current_cost�E�; ���+       ��K	N~z��A�>*

logging/current_costE��;U8'+       ��K	��z��A�>*

logging/current_costN-�;�N~�+       ��K	��z��A�>*

logging/current_cost�7�;vSL�+       ��K	P{��A�>*

logging/current_cost�]�;�*�7+       ��K	s<{��A�>*

logging/current_cost�[�;Dq��+       ��K	@l{��A�>*

logging/current_cost�6�;/��!+       ��K	��{��A�>*

logging/current_cost�-�;��#]+       ��K	,6|��A�>*

logging/current_cost�o�;���Z+       ��K	��|��A�>*

logging/current_cost$P�;E��"+       ��K		�|��A�>*

logging/current_cost�F�;]O�7+       ��K	4}��A�>*

logging/current_cost�;�;۷q+       ��K	zC}��A�>*

logging/current_costtM�;�m+       ��K	�}��A�>*

logging/current_costE�;j���+       ��K	�}��A�>*

logging/current_cost�P�;��+       ��K	x�}��A�?*

logging/current_cost�W�;��T+       ��K	�/~��A�?*

logging/current_cost �;>TRj+       ��K	Yf~��A�?*

logging/current_cost�o�;�"��+       ��K	��~��A�?*

logging/current_cost�[�;8�Y+       ��K	��~��A�?*

logging/current_cost�'�;T)��+       ��K		��A�?*

logging/current_cost<k�;#���+       ��K	�7��A�?*

logging/current_costE*�;���+       ��K	�o��A�?*

logging/current_costo�;L��+       ��K	���A�?*

logging/current_cost�-�;6��z+       ��K	���A�?*

logging/current_cost;i�;b���+       ��K	R���A�?*

logging/current_cost<R�;�L�+       ��K	�>���A�?*

logging/current_costu-�;����+       ��K	�p���A�?*

logging/current_cost�N�;<~�+       ��K	����A�?*

logging/current_cost$e�;t�FF+       ��K	�π��A�?*

logging/current_cost~4�;��+       ��K	����A�?*

logging/current_costr3�;��"+       ��K	�0���A�?*

logging/current_cost9^�;T�o+       ��K	�]���A�?*

logging/current_cost�h�;�-R7+       ��K	�����A�?*

logging/current_costK4�;��Ġ+       ��K	�����A�?*

logging/current_cost�B�;+�+       ��K	�쁸�A�?*

logging/current_cost�R�;=Oz�+       ��K	 ���A�?*

logging/current_cost�;�;�[,+       ��K	Y���A�?*

logging/current_costDg�;�T+       ��K	����A�?*

logging/current_costGT�;6�%+       ��K	т��A�?*

logging/current_costi$�;�/G+       ��K	l ���A�?*

logging/current_cost{m�;�B|+       ��K	�V���A�@*

logging/current_cost�K�;���{+       ��K	G����A�@*

logging/current_costrg�;�6�+       ��K	�̃��A�@*

logging/current_costD+�;��4+       ��K	�����A�@*

logging/current_cost�B�;]s�+       ��K	L4���A�@*

logging/current_cost`Y�;�P�+       ��K	*o���A�@*

logging/current_costWT�;sR&(+       ��K	�����A�@*

logging/current_cost�X�;4�$�+       ��K	�儸�A�@*

logging/current_costiY�;����+       ��K	m���A�@*

logging/current_cost�'�;��Q+       ��K	�L���A�@*

logging/current_cost�t�;��/+       ��K	�����A�@*

logging/current_cost7�; ]'t+       ��K	?����A�@*

logging/current_cost�h�;Ѡ�x+       ��K	�����A�@*

logging/current_cost��;���+       ��K	.���A�@*

logging/current_costr�;���,+       ��K	�]���A�@*

logging/current_costnF�;L�Nc+       ��K	����A�@*

logging/current_costR\�;E�{�+       ��K	�І��A�@*

logging/current_cost��;ѓ��+       ��K	>���A�@*

logging/current_cost ]�;��<?+       ��K	v8���A�@*

logging/current_cost�D�;��[�+       ��K	@h���A�@*

logging/current_cost�9�;�x�+       ��K	k����A�@*

logging/current_costK�;��~+       ��K	�ч��A�@*

logging/current_cost E�;��Sg+       ��K	4���A�@*

logging/current_cost�<�;�j}>+       ��K	�C���A�@*

logging/current_cost�A�;��D�+       ��K	Y}���A�@*

logging/current_cost�x�;+z�+       ��K	^����A�A*

logging/current_cost�C�;�t�+       ��K	.舸�A�A*

logging/current_cost�!�;�0�+       ��K	����A�A*

logging/current_cost�C�;���8+       ��K	�O���A�A*

logging/current_cost�\�;���.+       ��K	����A�A*

logging/current_cost�2�;օ\+       ��K	�É��A�A*

logging/current_costc�;(���+       ��K	p��A�A*

logging/current_cost�W�;jB+       ��K	�%���A�A*

logging/current_cost�&�;�b;+       ��K	>^���A�A*

logging/current_cost+F�;�
\�+       ��K	[����A�A*

logging/current_costel�;4���+       ��K	3܊��A�A*

logging/current_cost�3�;|��+       ��K	����A�A*

logging/current_cost�B�;�r25+       ��K	p@���A�A*

logging/current_cost�M�;c�+       ��K	�q���A�A*

logging/current_cost�<�;|L�+       ��K	آ���A�A*

logging/current_cost;^�;���;+       ��K	xӋ��A�A*

logging/current_cost�i�;���+       ��K	����A�A*

logging/current_costY,�;<w7]+       ��K	�4���A�A*

logging/current_costrP�;1i\+       ��K	�i���A�A*

logging/current_cost�5�;|�+       ��K	�����A�A*

logging/current_cost�P�;�x��+       ��K	.̌��A�A*

logging/current_cost�h�;>K�S+       ��K	o����A�A*

logging/current_cost`%�;�9C�+       ��K	9)���A�A*

logging/current_costn]�;���=+       ��K	�V���A�A*

logging/current_cost�:�;��h+       ��K	򊍸�A�A*

logging/current_coste\�;��[+       ��K	�����A�A*

logging/current_cost�L�;�rv�+       ��K	�荸�A�B*

logging/current_costKC�;�Y<�+       ��K	���A�B*

logging/current_cost9:�;��D�+       ��K	�H���A�B*

logging/current_cost�E�;Cxa+       ��K	�y���A�B*

logging/current_cost"R�;m5�b+       ��K	𩎸�A�B*

logging/current_cost�f�;+��+       ��K	�َ��A�B*

logging/current_cost^,�;^.'�+       ��K	+���A�B*

logging/current_cost|I�;�M+       ��K	4���A�B*

logging/current_costNo�;�L+       ��K	�c���A�B*

logging/current_cost|2�;_^+       ��K	*����A�B*

logging/current_cost+2�;s*C+       ��K	nя��A�B*

logging/current_cost�?�;�E�+       ��K	�����A�B*

logging/current_cost�M�;N�g�+       ��K	Z.���A�B*

logging/current_cost�k�;�_�+       ��K	�`���A�B*

logging/current_cost��;D+�+       ��K	����A�B*

logging/current_costُ�;�g��+       ��K	d͐��A�B*

logging/current_cost���;i%C[+       ��K	����A�B*

logging/current_costB��;���+       ��K	91���A�B*

logging/current_cost��;��׸+       ��K	`���A�B*

logging/current_costdM�;7� �+       ��K	(����A�B*

logging/current_cost�;u�4+       ��K	n����A�B*

logging/current_cost��;���f+       ��K	�푸�A�B*

logging/current_cost�P�;H��+       ��K	����A�B*

logging/current_cost�l�;GM�+       ��K	6J���A�B*

logging/current_cost�4�;��K�+       ��K	�x���A�B*

logging/current_cost�1�;j��_+       ��K	?����A�B*

logging/current_cost%��;���+       ��K	Vڒ��A�C*

logging/current_cost��;��>P+       ��K	S
���A�C*

logging/current_costB�;�+       ��K	�9���A�C*

logging/current_cost�f�;f��+       ��K	�m���A�C*

logging/current_cost5(�;�D��+       ��K	Q����A�C*

logging/current_cost�3�;�^� +       ��K	�Г��A�C*

logging/current_cost�'�;�"�8+       ��K	����A�C*

logging/current_costGK�;�G��+       ��K	�5���A�C*

logging/current_costr<�;;�x9+       ��K	�d���A�C*

logging/current_cost�P�;��j�+       ��K	n����A�C*

logging/current_cost�$�;�u�+       ��K	�����A�C*

logging/current_costu�;�R��+       ��K	�����A�C*

logging/current_cost@��;?k�V+       ��K	7(���A�C*

logging/current_cost"�;�`m�+       ��K	�X���A�C*

logging/current_cost�K�;��_�+       ��K	󈕸�A�C*

logging/current_cost�<�;��K�+       ��K	S����A�C*

logging/current_cost%,�;)Ku�+       ��K	1�A�C*

logging/current_cost�U�;�`k�+       ��K	+���A�C*

logging/current_costS�;��+       ��K	%M���A�C*

logging/current_cost|)�;�)�+       ��K	�|���A�C*

logging/current_costPc�;�1+       ��K	I����A�C*

logging/current_cost�6�;]L	�+       ��K	ؖ��A�C*

logging/current_cost�E�;�H+       ��K	���A�C*

logging/current_cost�p�;b��+       ��K	�C���A�C*

logging/current_cost%N�;{Q�+       ��K	�p���A�C*

logging/current_cost�>�;7�+       ��K	�����A�D*

logging/current_cost�@�;gNzk+       ��K	�җ��A�D*

logging/current_cost1�;$��+       ��K	����A�D*

logging/current_cost�V�;���+       ��K	�6���A�D*

logging/current_cost�V�;���+       ��K	xe���A�D*

logging/current_cost9�;I�4�+       ��K	H����A�D*

logging/current_costIf�;�+PD+       ��K	ɘ��A�D*

logging/current_cost�+�;b �+       ��K	U����A�D*

logging/current_cost�L�;�lR4+       ��K	L$���A�D*

logging/current_cost.2�;�9�I+       ��K	�Y���A�D*

logging/current_cost.m�;e_��+       ��K	/����A�D*

logging/current_cost�Q�;����+       ��K	ܹ���A�D*

logging/current_cost9V�;��m�+       ��K	�虸�A�D*

logging/current_cost@�;�m>�+       ��K	2���A�D*

logging/current_cost�H�;�m��+       ��K	GH���A�D*

logging/current_cost{p�;�<��+       ��K	�u���A�D*

logging/current_cost�5�;a"
�+       ��K	d����A�D*

logging/current_cost,`�;D�^2+       ��K	Rך��A�D*

logging/current_cost�R�;��`+       ��K	(���A�D*

logging/current_costuA�;a�A�+       ��K	�2���A�D*

logging/current_cost��;��r�+       ��K	�`���A�D*

logging/current_cost��;nr�+       ��K	�����A�D*

logging/current_cost�^�;v�K)+       ��K	½���A�D*

logging/current_cost�s�;il_l+       ��K	�꛸�A�D*

logging/current_cost�A�;۱n�+       ��K	���A�D*

logging/current_cost@�;_�X +       ��K	{G���A�D*

logging/current_costK<�;jr��+       ��K	zs���A�E*

logging/current_cost	��;�<y�+       ��K	�����A�E*

logging/current_cost�8�;��+       ��K	�Μ��A�E*

logging/current_costd �;h�;
+       ��K	+���A�E*

logging/current_cost"L�;�.q+       ��K	�1���A�E*

logging/current_cost�R�;:NW+       ��K	:`���A�E*

logging/current_cost�g�;-mz+       ��K	�����A�E*

logging/current_cost\'�;D>��+       ��K	����A�E*

logging/current_costPT�;ä^v+       ��K	�ꝸ�A�E*

logging/current_cost�[�;��+       ��K	����A�E*

logging/current_cost�u�;��+       ��K	�F���A�E*

logging/current_costN/�;9j�+       ��K	�t���A�E*

logging/current_cost��;A)��+       ��K	�����A�E*

logging/current_cost�j�;]�o�+       ��K	7͞��A�E*

logging/current_cost���;}=V�+       ��K	�����A�E*

logging/current_costB�;*���+       ��K	�)���A�E*

logging/current_cost|P�;��g+       ��K	�V���A�E*

logging/current_cost��;��s�+       ��K	7����A�E*

logging/current_cost\b�;��`<+       ��K	б���A�E*

logging/current_cost�H�;�o�+       ��K	�ߟ��A�E*

logging/current_cost7�;O���+       ��K	����A�E*

logging/current_cost�\�;�p��+       ��K	�<���A�E*

logging/current_cost�O�;��d+       ��K	Nn���A�E*

logging/current_cost��;����+       ��K	!����A�E*

logging/current_cost"��; �)�+       ��K	�Р��A�E*

logging/current_cost��;��3+       ��K	�����A�F*

logging/current_cost'��;S,�+       ��K	\,���A�F*

logging/current_cost��;r4��+       ��K	�Y���A�F*

logging/current_costN�;�rt3+       ��K	����A�F*

logging/current_cost�H�;7��+       ��K	����A�F*

logging/current_cost�^�;T��m+       ��K	桸�A�F*

logging/current_cost�D�;$�P+       ��K	R���A�F*

logging/current_cost��;�(j+       ��K	!C���A�F*

logging/current_cost �;
�8+       ��K	_o���A�F*

logging/current_cost56�;,H�*+       ��K	М���A�F*

logging/current_cost���;\���+       ��K	Т��A�F*

logging/current_cost��;� �t+       ��K	�����A�F*

logging/current_cost�L�;�'|+       ��K	]+���A�F*

logging/current_costd��;�+i�+       ��K	�Y���A�F*

logging/current_cost��;���1+       ��K	~����A�F*

logging/current_cost$~�;�oo+       ��K	t����A�F*

logging/current_cost`_�;I���+       ��K	*棸�A�F*

logging/current_cost���;%�W�+       ��K	����A�F*

logging/current_costrp�;ެ$�+       ��K	aC���A�F*

logging/current_cost��;)ra+       ��K	�r���A�F*

logging/current_costN�;wfY+       ��K	Ǟ���A�F*

logging/current_cost�S�; )1R+       ��K	�ͤ��A�F*

logging/current_cost�;���+       ��K	�����A�F*

logging/current_cost�m�;�F�+       ��K	�*���A�F*

logging/current_cost���;��+       ��K	3Y���A�F*

logging/current_cost��;��	+       ��K	u����A�F*

logging/current_cost2\�;q�V�+       ��K	����A�G*

logging/current_cost�\�;Q��&+       ��K	襸�A�G*

logging/current_cost�^�;��9+       ��K	���A�G*

logging/current_cost�V�;D��+       ��K	^J���A�G*

logging/current_cost�,�;3{j�+       ��K	|���A�G*

logging/current_cost�.�;g�'�+       ��K	n����A�G*

logging/current_cost��;��Q+       ��K	.զ��A�G*

logging/current_cost�C�;:�A$+       ��K	�
���A�G*

logging/current_costUW�;A\	5+       ��K	�9���A�G*

logging/current_cost�T�;Y��+       ��K	�e���A�G*

logging/current_cost�J�;�/��+       ��K	�����A�G*

logging/current_costu�;�⠣+       ��K	�§��A�G*

logging/current_costg;�;m|�+       ��K	)��A�G*

logging/current_costx�;��+       ��K	%"���A�G*

logging/current_cost2]�;[���+       ��K	�Q���A�G*

logging/current_cost�c�;�" +       ��K	G����A�G*

logging/current_cost<?�;E�7F+       ��K	𭨸�A�G*

logging/current_cost�6�;��
-+       ��K	�ܨ��A�G*

logging/current_cost�x�;��+       ��K	o
���A�G*

logging/current_cost�x�;2���+       ��K	E9���A�G*

logging/current_cost|�;�n�W+       ��K	�j���A�G*

logging/current_costi#�;ԃ�*+       ��K	�����A�G*

logging/current_cost���;�	k�+       ��K	GƩ��A�G*

logging/current_cost �;�՞�+       ��K	/��A�G*

logging/current_cost٪�;�d�+       ��K	L ���A�G*

logging/current_cost���;$�'+       ��K	�N���A�G*

logging/current_costŅ�;�{_+       ��K	�z���A�H*

logging/current_costل�;�;�a+       ��K	�����A�H*

logging/current_cost ��;�<��+       ��K	�Ԫ��A�H*

logging/current_cost���;���0+       ��K	���A�H*

logging/current_cost�1�;�ҥ+       ��K	A4���A�H*

logging/current_cost�"�;G���+       ��K	qb���A�H*

logging/current_cost~��;�{o+       ��K	�����A�H*

logging/current_costDl�;H���+       ��K	�����A�H*

logging/current_cost�!�;�!��+       ��K	�A�H*

logging/current_cost�g�;� j�+       ��K	����A�H*

logging/current_cost<��;���+       ��K	�I���A�H*

logging/current_cost�+�;�A�
+       ��K	fw���A�H*

logging/current_cost�7�;�,|�+       ��K	�����A�H*

logging/current_cost$��;E���+       ��K	�Ԭ��A�H*

logging/current_cost�#�;�0q+       ��K	E���A�H*

logging/current_cost|-�;��+ +       ��K	+1���A�H*

logging/current_cost>s�;n&�+       ��K	�]���A�H*

logging/current_costW�;E��W+       ��K	�����A�H*

logging/current_costU�;v�S+       ��K	7����A�H*

logging/current_cost�/�;�dA+       ��K	O�A�H*

logging/current_costW?�;ѡ!�+       ��K	����A�H*

logging/current_cost�v�;	�g�+       ��K	�J���A�H*

logging/current_cost2�;��h+       ��K	"z���A�H*

logging/current_costY$�;
̉�+       ��K	e����A�H*

logging/current_costN��;Dj��+       ��K	^ծ��A�H*

logging/current_cost���;EGX�+       ��K	����A�I*

logging/current_costT��;|.�+       ��K	�5���A�I*

logging/current_cost�G�;�_��+       ��K	�c���A�I*

logging/current_costk�;���"+       ��K	�����A�I*

logging/current_cost"8�;.] �+       ��K	�¯��A�I*

logging/current_cost���;�v�8+       ��K	��A�I*

logging/current_costiS�;����+       ��K	R#���A�I*

logging/current_costuA�;��ۏ+       ��K	R���A�I*

logging/current_cost�.�;*�+=+       ��K	ހ���A�I*

logging/current_costR��;	�W�+       ��K	®���A�I*

logging/current_costx�;���F+       ��K	�۰��A�I*

logging/current_cost Y�;yO��+       ��K	���A�I*

logging/current_cost5�;��rc+       ��K	1D���A�I*

logging/current_costI?�;��}+       ��K	�t���A�I*

logging/current_cost�w�;d��+       ��K	����A�I*

logging/current_cost]�;��+       ��K	Xر��A�I*

logging/current_cost�K�;��(+       ��K	J���A�I*

logging/current_cost.u�;���a+       ��K	�?���A�I*

logging/current_cost�-�;��s+       ��K	Po���A�I*

logging/current_costN.�;�c�+       ��K	r����A�I*

logging/current_costkq�;���+       ��K	(ٲ��A�I*

logging/current_costGD�;z�,�+       ��K	�	���A�I*

logging/current_cost ��;�s�D+       ��K	';���A�I*

logging/current_cost^��;l�+       ��K	�q���A�I*

logging/current_costY �;���+       ��K	�����A�I*

logging/current_cost+H�;�4�+       ��K	�ҳ��A�I*

logging/current_cost9��;�2q+       ��K	����A�J*

logging/current_cost0�;q}��+       ��K	�5���A�J*

logging/current_costd&�;Iz��+       ��K	�i���A�J*

logging/current_cost ��;4��+       ��K	Z����A�J*

logging/current_cost��;�f#�+       ��K	�Ǵ��A�J*

logging/current_cost9��;]���+       ��K	 ����A�J*

logging/current_cost�V�;kj�+       ��K	�&���A�J*

logging/current_cost f�;�T�+       ��K	�V���A�J*

logging/current_costk:�;է�+       ��K	Y����A�J*

logging/current_cost ��;�m'-+       ��K	�����A�J*

logging/current_cost�q�;7�C�+       ��K		嵸�A�J*

logging/current_cost�3�;��ɢ+       ��K	����A�J*

logging/current_costn'�;��q+       ��K	_A���A�J*

logging/current_cost���;"O7+       ��K	�o���A�J*

logging/current_costtp�;%�t�+       ��K	Ý���A�J*

logging/current_cost��;%6z�+       ��K	^˶��A�J*

logging/current_cost�V�;r��+       ��K	�����A�J*

logging/current_cost��;���d+       ��K	�&���A�J*

logging/current_cost���;��$�+       ��K	�S���A�J*

logging/current_cost'��;8�7�+       ��K	ȁ���A�J*

logging/current_costa�;�_��+       ��K	D����A�J*

logging/current_costˀ�;ɡ�+       ��K	�ܷ��A�J*

logging/current_cost���;�wC�+       ��K	�
���A�J*

logging/current_cost.2�;iF��+       ��K	�8���A�J*

logging/current_costR4�;�c�t+       ��K	df���A�J*

logging/current_coste}�;Z�B+       ��K	�����A�K*

logging/current_cost�;P>�+       ��K	'¸��A�K*

logging/current_cost�i�;��+       ��K	��A�K*

logging/current_cost9B�;�$��+       ��K	����A�K*

logging/current_cost�^�;��$+       ��K	�J���A�K*

logging/current_cost�^�;ɿ�+       ��K	�y���A�K*

logging/current_cost5�;ly��+       ��K	{����A�K*

logging/current_cost�x�;�++       ��K	�ӹ��A�K*

logging/current_cost�	�;fh�6+       ��K	^���A�K*

logging/current_cost�P�;벃�+       ��K	5.���A�K*

logging/current_cost9h�;]�Vx+       ��K	^���A�K*

logging/current_cost�_�;$�+       ��K	ϋ���A�K*

logging/current_costn�; ���+       ��K	>̺��A�K*

logging/current_cost�h�;�(V�+       ��K	?���A�K*

logging/current_cost:�;є��+       ��K	�B���A�K*

logging/current_cost��;̝I3+       ��K	p
���A�K*

logging/current_cost��;O9)'+       ��K	�Z���A�K*

logging/current_cost�w�;�'#7+       ��K	�ڼ��A�K*

logging/current_cost�B�;���3+       ��K	����A�K*

logging/current_costU�;7m�+       ��K	�X���A�K*

logging/current_cost�z�;%�+       ��K	�P���A�K*

logging/current_costd@�;�/0�+       ��K	�þ��A�K*

logging/current_costb�;5H�3+       ��K	����A�K*

logging/current_cost�?�;6�yy+       ��K	~F���A�K*

logging/current_cost>A�;�j��+       ��K	�{���A�K*

logging/current_costb�;|NIt+       ��K	�����A�K*

logging/current_cost|H�;@��>+       ��K	W޿��A�L*

logging/current_cost�T�;{��+       ��K	����A�L*

logging/current_cost|T�;)���+       ��K	�L���A�L*

logging/current_costgm�;:�C�+       ��K	�����A�L*

logging/current_cost+��;��x+       ��K	ض���A�L*

logging/current_cost`@�;X^�+       ��K	[����A�L*

logging/current_cost~ �;�}�+       ��K	����A�L*

logging/current_cost� �;�%��+       ��K	6W���A�L*

logging/current_cost@��;}%�+       ��K	؊���A�L*

logging/current_cost���;���+       ��K	�����A�L*

logging/current_cost�i�;.��+       ��K	-¸�A�L*

logging/current_costܖ�;� �~+       ��K	�:¸�A�L*

logging/current_costg.�;��;�+       ��K	.m¸�A�L*

logging/current_coste��;��t�+       ��K	أ¸�A�L*

logging/current_costr[�;��'�+       ��K	��¸�A�L*

logging/current_costr��;}�C=+       ��K	3
ø�A�L*

logging/current_costE��;	��+       ��K	�>ø�A�L*

logging/current_cost92�; �ƹ+       ��K	�nø�A�L*

logging/current_cost�;x��Q+       ��K	�ø�A�L*

logging/current_costǀ�;�B��+       ��K	q�ø�A�L*

logging/current_cost F�;�z�+       ��K	��ø�A�L*

logging/current_cost��;��zc+       ��K	y0ĸ�A�L*

logging/current_costg%�;і�+       ��K	*cĸ�A�L*

logging/current_costՠ�;5�+       ��K	��ĸ�A�L*

logging/current_cost�%�;Ko��+       ��K	��ĸ�A�L*

logging/current_cost�7�;�U��+       ��K	8 Ÿ�A�L*

logging/current_cost��;-E3v+       ��K	x4Ÿ�A�M*

logging/current_cost��;l���+       ��K	iŸ�A�M*

logging/current_cost~_�;��X+       ��K	b�Ÿ�A�M*

logging/current_cost�_�;;((�+       ��K	�Ÿ�A�M*

logging/current_cost`�;ǆ�a+       ��K	|�Ÿ�A�M*

logging/current_cost�y�;���$+       ��K	z(Ƹ�A�M*

logging/current_cost�c�;y!n1+       ��K	VƸ�A�M*

logging/current_cost0��;Q��+       ��K	��Ƹ�A�M*

logging/current_cost�L�;v��+       ��K	w�Ƹ�A�M*

logging/current_cost�u�;�I+       ��K	��Ƹ�A�M*

logging/current_cost�;�:u%+       ��K	�Ǹ�A�M*

logging/current_cost���;�o��+       ��K	8KǸ�A�M*

logging/current_costr��;2۽E+       ��K	�xǸ�A�M*

logging/current_costTO�;$��+       ��K	y�Ǹ�A�M*

logging/current_cost�E�;#�
"+       ��K	n�Ǹ�A�M*

logging/current_cost��;��?�+       ��K	8ȸ�A�M*

logging/current_cost�x�;e�|�+       ��K	L=ȸ�A�M*

logging/current_cost��;��þ+       ��K	atȸ�A�M*

logging/current_costG�;��+       ��K	��ȸ�A�M*

logging/current_cost�$�;ý>�+       ��K	X�ȸ�A�M*

logging/current_cost{/�;l+       ��K	aɸ�A�M*

logging/current_cost	��;����+       ��K	�9ɸ�A�M*

logging/current_cost �;%���+       ��K	�pɸ�A�M*

logging/current_cost�W�;���G+       ��K	K�ɸ�A�M*

logging/current_cost">�;�4�+       ��K	��ɸ�A�M*

logging/current_costR��;DA+       ��K	� ʸ�A�N*

logging/current_cost<��;��l�+       ��K	�0ʸ�A�N*

logging/current_cost���;�6 �+       ��K	�pʸ�A�N*

logging/current_cost�;��F�